import atexit
import itertools
import re
import gc
import statistics
import timeit
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, FrozenSet

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn
from stanza.server import CoreNLPClient
import torch
from sentence_transformers import SentenceTransformer, util
from fastcoref import FCoref

from rq1.metrics.utils.tensor2attr import Tensor2Attr

####################################################################################################
# Data Classes and Base Components
####################################################################################################

@dataclass
class MegaHRWord:
    """MegaHR dictionary word entry"""
    word: str
    conc: float
    imag: float

@dataclass
class MRCWord:
    """MRC Psycholinguistic Database word entry"""
    word: str
    wtype: str     # Part of speech
    conc: float    # Concreteness (0-700 scale)
    imag: float    # Imageability (0-700 scale)
    fami: float    # Familiarity (0-700 scale)

@dataclass
class DocumentStructure:
    """Container for processed document structure"""
    paragraphs: List[str]
    sentences: Dict[str, List[str]]
    tokens: pd.DataFrame
    token_embeddings: Dict[int, Any]
    sentence_count: int
    paragraph_count: int

####################################################################################################
# Helper Classes
####################################################################################################

class MegaHRDictionary:
    """Helper class for MegaHR dictionary operations"""
    def __init__(self, dictionary_path: str):
        self.dictionary = self._load_dictionary(dictionary_path)

    def _load_dictionary(self, dictionary_path: str) -> Dict[str, MegaHRWord]:
        words = {}
        with open(dictionary_path, 'r', encoding='utf-8-sig') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    words[parts[0].lower()] = MegaHRWord(
                        word=parts[0].lower(),
                        conc=float(parts[1]),
                        imag=float(parts[2])
                    )
        return words

    def find_word(self, word: str) -> Optional[MegaHRWord]:
        return self.dictionary.get(word.lower())

class MRCDatabase:
    """Interface to the MRC Psycholinguistic Database"""
    def __init__(self, csv_path: str):
        self.pos_mapping = self._get_spacy_to_mrc_pos_mapping()

        # Load from local CSV file instead of Hugging Face
        self.df = pd.read_csv(
            csv_path,
            keep_default_na=False,
            na_values=[""]
        )
        
        self.df['Word'] = self.df['Word'].str.lower()
        
        # Convert numeric columns
        for col in ['Concreteness', 'Imageability', 'Familiarity']:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _get_spacy_to_mrc_pos_mapping(self) -> Dict[str, List[str]]:
        return {
            'NOUN': ['N'],
            'PROPN': ['N'],
            'ADJ': ['J'],
            'VERB': ['V', 'P'],
            'ADV': ['A'],
            'ADP': ['A', 'R', 'C'],
            'DET': ['U'],
            'PRON': ['U'],
            'CCONJ': ['C'],
            'INTJ': ['I']
        }

    def find_word(self, word: str, pos: str) -> List[MRCWord]:
        mrc_pos = self.pos_mapping.get(pos, ['O'])
        matches = self.df[
            (self.df['Word'] == word.lower()) & 
            (self.df['Comprehensive Syntactic Category'].isin(mrc_pos))
        ]
        
        if len(matches) == 0:
            matches = self.df[self.df['Word'] == word.lower()]
        
        return [
            MRCWord(
                word=row['Word'],
                wtype=row['Comprehensive Syntactic Category'],
                conc=float(row['Concreteness']) if pd.notna(row['Concreteness']) else 0.0,
                imag=float(row['Imageability']) if pd.notna(row['Imageability']) else 0.0,
                fami=float(row['Familiarity']) if pd.notna(row['Familiarity']) else 0.0
            )
            for _, row in matches.iterrows()
        ]

class TextProcessor:
    """Helper class for text processing"""
    def __init__(self, model_name='en_core_web_trf'):
        """Initialize with spaCy model name"""    
        spacy.prefer_gpu()  # Use GPU if available
    
        # Load spaCy model directly
        self.nlp = spacy.load(model_name, disable={'ner', 'parser'})

        if 'tensor2attr' not in self.nlp.pipe_names:
            self.nlp.add_pipe('tensor2attr')
            
        # Add sentencizer if not already in pipeline
        if 'sentencizer' not in self.nlp.pipe_names:
            sentencizer_position = 'before' if 'ner' in self.nlp.pipe_names else 'last'
            position_arg = {'before': 'ner'} if sentencizer_position == 'before' else None
            self.nlp.add_pipe('sentencizer', **position_arg if position_arg else {})

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.encode("ascii", errors="replace").decode()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()

    def process_document(self, text: str) -> DocumentStructure:
        """Process a document to extract its structure"""
        text = self.clean_text(text)
        if not text:
            # Return empty structure for empty text
            return DocumentStructure(
                paragraphs=[],
                sentences={},
                tokens=pd.DataFrame(columns=["u_id", "p_id", "s_id", "token_id", 
                                           "token_text", "token_lemma", "token_pos"]),
                token_embeddings={},
                sentence_count=0,
                paragraph_count=0
            )
            
        paragraphs = text.split('\n')
        
        df_rows = []
        token_embeddings = {}
        sentences = {}
        
        p_id = 0
        u_id = 0
        total_sentences = 0

        # Process paragraphs in batches to reduce memory pressure
        batch_size = 10  # Adjust based on typical document size
        for batch_start in range(0, len(paragraphs), batch_size):
            batch_end = min(batch_start + batch_size, len(paragraphs))
            batch = paragraphs[batch_start:batch_end]
            
            for doc in self.nlp.pipe(batch, disable=["parser"]):
                sentences[str(p_id)] = []
                
                s_id = 0
                for sent in doc.sents:
                    t_id = 0
                    sentence_tokens = []
                    
                    for token in sent:
                        if token.text.strip():  # Skip empty tokens
                            df_rows.append({
                                "u_id": u_id,
                                "p_id": p_id,
                                "s_id": s_id,
                                "token_id": t_id,
                                "token_text": token.text.strip(),
                                "token_lemma": token.lemma_.strip(),
                                "token_pos": token.pos_
                            })

                            token_embeddings[u_id] = token.vector
                            sentence_tokens.append(token.text)

                            u_id += 1
                            t_id += 1

                    if sentence_tokens:  # Skip empty sentences
                        sentences[str(p_id)].append(' '.join(sentence_tokens))
                        s_id += 1
                        total_sentences += 1
                
                p_id += 1

        df_doc = pd.DataFrame(df_rows)

        return DocumentStructure(
            paragraphs=paragraphs,
            sentences=sentences,
            tokens=df_doc,
            token_embeddings=token_embeddings,
            sentence_count=total_sentences,
            paragraph_count=p_id
        )

    def get_token_ids_by_sentence(self, doc_structure: DocumentStructure) -> Dict[str, List[int]]:
        """Get token IDs organized by sentence"""
        sentences_tokens = {}
        df_doc = doc_structure.tokens
        
        if df_doc.empty:
            return {}
            
        # Group by paragraph and sentence ID in a vectorized way
        for p_id in df_doc['p_id'].unique():
            df_paragraph = df_doc.loc[df_doc['p_id'] == p_id]
            
            for s_id in df_paragraph['s_id'].unique():
                sentence_tokens = df_paragraph.loc[
                    df_paragraph['s_id'] == s_id, 'u_id'
                ].tolist()
                sentences_tokens[f'{p_id}_{s_id}'] = sentence_tokens
                
        return sentences_tokens

    def filter_tokens_by_pos(self, doc_structure: DocumentStructure, 
                        token_ids_by_sentence: Dict[str, List[int]],
                        pos_tags: List[str]) -> Dict[str, List[List[Dict[str, Any]]]]:
        """Filter tokens by POS tags and get their embeddings, organized by paragraph and sentence."""
        tokens = {}
        df_doc = doc_structure.tokens
        
        if df_doc.empty:
            return {}
            
        # Get all relevant tokens in one go
        relevant_tokens = df_doc[df_doc['token_pos'].isin(pos_tags)]
        
        for p_s_id, token_ids in token_ids_by_sentence.items():
            p_id = p_s_id.split('_')[0]
            
            if p_id not in tokens:
                tokens[p_id] = []
                
            current_tokens = []
            # Filter to tokens that are in this sentence and have relevant POS
            sentence_tokens = relevant_tokens[relevant_tokens['u_id'].isin(token_ids)]
            
            for _, row in sentence_tokens.iterrows():
                u_id = row['u_id']
                token_text = row['token_text']
                pos_tag = row['token_pos']
                
                # Use specific POS tag for WordNet as in original code
                synsets = set()
                if pos_tag == 'VERB':
                    synsets = set(wn.synsets(token_text, wn.VERB))
                elif pos_tag in ['NOUN', 'PROPN']:
                    synsets = set(wn.synsets(token_text, wn.NOUN))
                elif pos_tag == 'ADJ':
                    synsets = set(wn.synsets(token_text, wn.ADJ))
                elif pos_tag == 'ADV':
                    synsets = set(wn.synsets(token_text, wn.ADV))
                
                current_tokens.append({
                    'id': u_id,
                    'text': token_text,
                    'embedding': doc_structure.token_embeddings[u_id],
                    'synsets': synsets
                })
                    
            if current_tokens:
                tokens[p_id].append(current_tokens)
                
        return tokens

class EmbeddingsHelper:
    """Helper class for computing embeddings and similarities"""
    def __init__(self, model_name: str = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'):
        self.model = None  # Lazy loading
        self.model_name = model_name

    def _ensure_model_loaded(self):
        """Ensure the model is loaded when needed"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def compute_sentence_embeddings(self, sentences: List[str]) -> List[np.ndarray]:
        """Compute embeddings for a list of sentences in batch"""
        if not sentences:
            return []
            
        self._ensure_model_loaded()
        
        # Process in smaller batches to reduce memory pressure
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings

    def compute_local_cosine(self, embeddings: List[np.ndarray]) -> float:
        """Compute cosine similarity between consecutive embeddings"""
        if len(embeddings) <= 1:
            return 0.0
            
        scores = []
        for i in range(len(embeddings) - 1):
            scores.append(util.cos_sim(embeddings[i], embeddings[i + 1]).item())
        return statistics.mean(scores) if scores else 0.0

    def compute_global_cosine(self, embeddings: List[np.ndarray]) -> float:
        """Compute cosine similarity between all pairs of embeddings"""
        if len(embeddings) <= 1:
            return 0.0
            
        scores = []
        for pair in itertools.combinations(embeddings, r=2):
            scores.append(util.cos_sim(pair[0], pair[1]).item())

        return statistics.mean(scores) if scores else 0.0

    @lru_cache(maxsize=200000)
    def _compute_token_similarity(self, token1_id: int, token2_id: int, 
                                 embedding1_tuple: Tuple[float, ...], 
                                 embedding2_tuple: Tuple[float, ...]) -> float:
        """Compute similarity between token pairs with caching"""
        # Convert tuples back to numpy arrays
        embedding1 = np.array(embedding1_tuple)
        embedding2 = np.array(embedding2_tuple)
        
        # Compute similarity
        return util.cos_sim(embedding1, embedding2).item()

    def compute_token_similarities(self, token_embeddings: Dict[int, np.ndarray], 
                                token_pairs: List[Tuple[int, int]]) -> Dict[str, float]:
        """Compute and cache similarities between token pairs"""
        similarities = {}
        for token1_id, token2_id in token_pairs:
            # Sort IDs to ensure consistent cache keys
            if token1_id > token2_id:
                token1_id, token2_id = token2_id, token1_id
            
            # Get embeddings and convert to hashable tuples
            embedding1_tuple = tuple(token_embeddings[token1_id].tolist())
            embedding2_tuple = tuple(token_embeddings[token2_id].tolist())
            
            # Use the cached method
            similarity = self._compute_token_similarity(token1_id, token2_id, 
                                                     embedding1_tuple, embedding2_tuple)
            
            # Store with consistent key format
            pair_id = f"{token1_id}@{token2_id}"
            similarities[pair_id] = similarity
            
        return similarities
    
class WordNetHelper:
    """Helper class for WordNet operations"""
    def __init__(self):
        self.similarity_functions = {
            'path': wn.path_similarity,
            'lch': wn.lch_similarity,
            'wup': wn.wup_similarity
        }

    @lru_cache(maxsize=100000)
    def _compute_synset_pair_similarity_inner(self, token_a: str, token_b: str, 
                                            synsets_a_frozen: FrozenSet, 
                                            synsets_b_frozen: FrozenSet) -> Dict[str, float]:
        """Inner function to compute similarity between synset pairs with caching"""
        # Convert frozensets back to regular sets
        synsets_a = set(synsets_a_frozen)
        synsets_b = set(synsets_b_frozen)
        
        scores = {
            'path': [],
            'lch': [],
            'wup': [],
            'binary': []
        }

        # Use the original sets for calculation
        binary = 1 if len(synsets_a.intersection(synsets_b)) > 0 else 0
        scores['binary'] = [binary]

        # process all combinations
        for synset_a, synset_b in itertools.product(synsets_a, synsets_b):
            for score_name, score_func in self.similarity_functions.items():
                score = score_func(synset_a, synset_b)
                if score is not None:
                    scores[score_name].append(score)

        result = {k: statistics.mean(v) if v else 0 for k, v in scores.items()}
        return result

    def compute_synset_pair_similarity(self, pair: Tuple[Tuple[str, Set], Tuple[str, Set]]) -> Dict[str, float]:
        """Wrapper for the cached inner function"""
        token_a, synsets_a = pair[0]
        token_b, synsets_b = pair[1]
        
        # Convert sets to frozensets to make them hashable
        synsets_a_frozen = frozenset(synsets_a)
        synsets_b_frozen = frozenset(synsets_b)
        
        # Call the cached inner function
        return self._compute_synset_pair_similarity_inner(token_a, token_b, 
                                                      synsets_a_frozen, synsets_b_frozen)

    @lru_cache(maxsize=10000)
    def compute_hypernymy_score(self, word: str, pos: str) -> float:
        """Compute hypernymy score with caching"""
        pos_tag = wn.VERB if pos == 'VERB' else wn.NOUN
        synsets = list(set(wn.synsets(word, pos_tag)))
        
        if not synsets:
            return 0.0

        hypernym_lengths = []
        # Use all synsets - no limit
        for synset in synsets:
            path_lengths = [len(path) for path in synset.hypernym_paths()]
            if path_lengths:
                hypernym_lengths.append(statistics.mean(path_lengths))

        result = statistics.mean(hypernym_lengths) if hypernym_lengths else 0.0
        return result

class CausalPatternHelper:
    """Helper class for detecting causal patterns without conflicts"""
    def __init__(self):
        # Organize patterns by category
        self._create_pattern_groups()
        # Compile all patterns
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def _create_pattern_groups(self):
        """Create pattern groups organized by category"""
        # Lead and rise patterns
        motion_patterns = [
            r"(.+?) (?P<keyword>lead to|leads to|led to|leading to) (.+)",
            r"(.+?) (?P<keyword>give rise to|gave rise to|given rise to|giving rise to) (.+)",
        ]
        
        # Direct causation patterns
        causation_patterns = [
            r"(.+?) (?P<keyword>induces|induced|inducing|induce) (.+)",
            r"(.+?) (?P<keyword>caused by|caused|causes|causing|cause) (.+)",
            r"(.+?) (?P<keyword>bring on|brought on|bringing on|brings on) (.+)",
            r"(.+?) (?P<keyword>result from|resulting from|results from|resulted from) (.+)",
        ]
        
        # Reason patterns
        reason_patterns = [
            r"(?P<keyword>the reason for) (.+?) (?:is|are|was|were) (.+)",
            r"(?P<keyword>the reasons for) (.+?) (?:is|are|was|were) (.+)",
            r"(?P<keyword>the reason of) (.+?) (?:is|are|was|were) (.+)",
            r"(?P<keyword>the reasons of) (.+?) (?:is|are|was|were) (.+)",
            r"(?:a|an|the|one) effect of (.+?) (?:is|are|was|were) (?P<keyword>.+)",
            r"(.+?) (?:is|are|was|were) (?P<keyword>a reason for|an reason for|the reason for|one reason for) (.+)",
            r"(.+?) (?:is|are|was|were) (?P<keyword>a reasons for|an reasons for|the reasons for|one reasons for) (.+)",
            r"(.+?) (?:is|are|was|were) (?P<keyword>a reason of|an reason of|the reason of|one reason of) (.+)",
            r"(.+?) (?:is|are|was|were) (?P<keyword>a reasons of|an reasons of|the reasons of|one reasons of) (.+)",
        ]
        
        # Conditional patterns
        conditional_patterns = [
            r"if (.+?), (?P<keyword>then) (.+)",
            r"if (.+?), (?P<keyword>,) (.+)",  # Comma as implicit causal link
        ]
        
        # Because patterns (specific before general)
        because_patterns = [
            r"(.+?) (?P<keyword>because of) (.+)",
            r"(?P<keyword>because) (.+?), (.+)",
            r"(.+?), (?P<keyword>because) (.+)",
            r"(.+?) (?P<keyword>because) (.+)",
        ]
        
        # Consequence patterns
        consequence_patterns = [
            r"(.+?),? (?P<keyword>thus|therefore|hence|consequently) (.+)",
            r"(.+?), (?P<keyword>as a consequence) (.+)",
            r"(?P<keyword>inasmuch as) (.+?), (.+)",
            r"(.+?), (?P<keyword>inasmuch as) (.+)",
            r"(?P<keyword>in consequence of) (.+?), (.+)",
            r"(.+?) (?P<keyword>in consequence of) (.+)",
            r"(?P<keyword>due to) (.+?), (.+)",
            r"(.+?) (?P<keyword>due to) (.+)",
            r"(?P<keyword>owing to) (.+?), (.+)",
            r"(.+?) (?P<keyword>owing to) (.+)",
            r"(.+?) (?P<keyword>as a result of) (.+)",
            r"(.+?) (?P<keyword>and hence) (.+)",
            r"(?P<keyword>as a consequence of) (.+?), (.+)",
            r"(.+?) (?P<keyword>as a consequence of) (.+)",
            r"(.+?) (?P<keyword>and consequently) (.+)",
            r"(.+?), (?P<keyword>for this reason alone) (.+)",
        ]
        
        # Combine all patterns
        self.patterns = (
            motion_patterns + 
            causation_patterns + 
            reason_patterns + 
            conditional_patterns + 
            because_patterns + 
            consequence_patterns
        )

    def count_causal_patterns(self, text: str) -> int:
        """Count causal patterns in text, avoiding overlapping keyword matches"""
        count = 0
        used_positions = set()  # Track positions of causal keywords only
        
        for pattern in self.compiled_patterns:
            for match in pattern.finditer(text.lower()):
                # Get the span of the named group 'keyword'
                keyword_span = match.span("keyword")
                start, end = keyword_span
                    
                # Check for overlap with previously used positions
                overlap = False
                for pos in range(start, end):
                    if pos in used_positions:
                        overlap = True
                        break
                    
                if not overlap:
                    # Mark only the keyword positions as used
                    for pos in range(start, end):
                        used_positions.add(pos)
                    count += 1

        return count

####################################################################################################
# Core GisPy Implementation
####################################################################################################

class GisPy:
    """Main class for computing Gist Inference Score components"""
    
    def __init__(self, 
                 model_name='en_core_web_trf',
                 megahr_path=None, 
                 mrc_path=None, 
                 use_corenlp=False,
                 corenlp_memory='4G',
                 sentence_model='kamalkraj/BioSimCSE-BioLinkBERT-BASE',
                 enabled_metrics=None):
        """Initialize all helper components with configurable paths and settings"""
        
        # Configure which metrics to enable
        self._init_metrics(enabled_metrics)
        
        # Initialize text processor with model_name
        self.text_processor = TextProcessor(model_name)

        self.embeddings_helper = EmbeddingsHelper(sentence_model)
        self.wordnet_helper = WordNetHelper()
        self.causal_helper = CausalPatternHelper()
        
        # Initialize dictionaries if paths are provided
        self.mrc_db = MRCDatabase(mrc_path) if mrc_path else None
        self.megahr_dict = MegaHRDictionary(megahr_path) if megahr_path else None

        # Configure coref options
        self.use_corenlp = use_corenlp and self.enabled_groups.get('coref', False)
        self.use_fastcoref = not self.use_corenlp and self.enabled_groups.get('coref', False)
        
        # CoreNLP setup (only if explicitly chosen)
        self.client = None
        self.corenlp_memory = corenlp_memory
        
        # Initialize fastcoref (lazy loading)
        self.coref_model = None

    def _ensure_corenlp_client(self):
        if self.use_corenlp and not self.client:
            self.client = CoreNLPClient(
                annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'coref'],
                endpoint='http://localhost:9017',
                threads=10,
                timeout=3000000,
                memory=self.corenlp_memory,
                be_quiet=True,
            )
            self.client.__enter__()
            atexit.register(self._cleanup_corenlp)
            
    def _ensure_fastcoref(self):
        """Ensure fastcoref model is loaded when needed"""
        if self.use_fastcoref and self.coref_model is None:
            print("Initializing fastcoref model...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.coref_model = FCoref(device=device)
            print(f"FastCoref initialized on {device}")

    def _init_metrics(self, enabled_metrics=None):
        """Configure which metrics to calculate."""
        # Default settings - enable all
        self.enabled_groups = {
            'coref': True,          # Coreference resolution
            'pcref': True,          # Referential cohesion
            'pcdc': True,           # Deep cohesion
            'causality': True,      # Causality metrics
            'concreteness': True,   # Concreteness metrics
            'wordnet': True,        # WordNet-based metrics
        }
        
        # Override with user-specified enabled metrics if provided
        if enabled_metrics is not None:
            for group in self.enabled_groups:
                if group in enabled_metrics:
                    self.enabled_groups[group] = enabled_metrics[group]
        
        # Log enabled metrics
        enabled_groups = [g for g, e in self.enabled_groups.items() if e]
        print(f"GisPy enabled metric groups: {enabled_groups}")

    def _cleanup_corenlp(self):
        """Clean up CoreNLP client properly"""
        if hasattr(self, 'client') and self.client:
            self.client.__exit__(None, None, None)
            self.client = None

    def __del__(self):
        """Ensure proper cleanup"""
        self._cleanup_corenlp()
        
    def compute_indices(self, text: str) -> Dict[str, float]:
        """Compute all indices needed for GIS calculation"""
        if not text:
            return self._get_default_indices()
                
        # Process document
        doc = self.text_processor.process_document(text)
        token_ids_by_sentence = self.text_processor.get_token_ids_by_sentence(doc)

        # Initialize with default values for all indices
        indices = self._get_default_indices()
        
        # Compute enabled metrics
        metric_functions = {
            'coref': self._compute_coref_metrics,
            'pcref': self._compute_pcref_metrics,
            'pcdc': self._compute_pcdc_metrics,
        }
        
        # Special case for token-based metrics that are computed together
        token_based_needed = any(self.enabled_groups[group] for group in ['causality', 'concreteness', 'wordnet'])
        
        # Compute each enabled metric group
        for metric_group, compute_fn in metric_functions.items():
            if self.enabled_groups[metric_group]:
                indices.update(compute_fn(doc))
        
        # Compute token-based metrics if any are enabled
        if token_based_needed:
            indices.update(self._compute_token_based_indices(doc, token_ids_by_sentence))

        return indices

    def _compute_coref_metrics(self, doc: DocumentStructure) -> Dict[str, float]:
        """Compute coreference metrics using either fastcoref (default) or CoreNLP"""
        if not self.enabled_groups.get('coref', False):
            return {'CoREF': 0.0}
            
        if self.use_corenlp:
            # Use CoreNLP for coref
            self._ensure_corenlp_client()
            
            # Prepare paragraphs for processing
            paragraph_texts = []
            for p_id, sentences in doc.sentences.items():
                if sentences:
                    paragraph_texts.append(' '.join(sentences))
            
            # Process paragraphs sequentially
            coref_scores = []
            for text in paragraph_texts:
                if not text:
                    coref_scores.append(0.0)
                    continue
                
                ann = self.client.annotate(text)
                chain_count = len(list(ann.corefChain))
                sentence_count = len(list(ann.sentence))
                score = chain_count / sentence_count if sentence_count else 0.0
                coref_scores.append(score)
            
            return {'CoREF': statistics.mean(coref_scores) if coref_scores else 0.0}
            
        elif self.use_fastcoref:
            # Use fastcoref for coref (default)
            self._ensure_fastcoref()
            
            coref_scores = []
            for p_id, sentences in doc.sentences.items():
                if not sentences:
                    continue
                    
                paragraph_text = ' '.join(sentences)
                if not paragraph_text:
                    continue
                
                # Process with fastcoref
                prediction = self.coref_model.predict(texts=[paragraph_text])[0]
                
                # Get clusters from prediction
                clusters = prediction.get_clusters()
                cluster_count = len(clusters)
                sentence_count = len(sentences)
                
                if sentence_count:
                    score = cluster_count / sentence_count
                    coref_scores.append(score)
            
            return {'CoREF': statistics.mean(coref_scores) if coref_scores else 0.0}
        
        return {'CoREF': 0.0}

    def _compute_pcref_metrics(self, doc: DocumentStructure) -> Dict[str, float]:
        """Compute referential cohesion metrics if enabled"""
        # Get sentence embeddings
        all_sentences = [sentence for sentences in doc.sentences.values() for sentence in sentences]
        all_embeddings = self.embeddings_helper.compute_sentence_embeddings(all_sentences)

        # Map embeddings back to paragraph structure
        sentence_embeddings = {}
        s_index = 0
        
        for p_id, sentences in doc.sentences.items():
            sentence_embeddings[p_id] = []
            for _ in sentences:
                if s_index < len(all_embeddings):
                    sentence_embeddings[p_id].append(all_embeddings[s_index])
                    s_index += 1

        # Compute referential cohesion scores
        return self._compute_referential_cohesion(sentence_embeddings)

    def _compute_pcdc_metrics(self, doc: DocumentStructure) -> Dict[str, float]:
        """Compute deep cohesion metrics if enabled"""
        return {'PCDC': self._compute_deep_cohesion(doc)}

    def _get_default_indices(self) -> Dict[str, float]:
        """Return default zero values for all indices"""
        return {
            'CoREF': 0.0,
            'PCREF_1': 0.0, 'PCREF_a': 0.0, 'PCREF_1p': 0.0, 'PCREF_ap': 0.0,
            'PCDC': 0.0,
            'SMCAUSe_1': 0.0, 'SMCAUSe_a': 0.0, 'SMCAUSe_1p': 0.0, 'SMCAUSe_ap': 0.0,
            'PCCNC_megahr': 0.0, 'WRDIMGc_megahr': 0.0,
            'PCCNC_mrc': 0.0, 'WRDIMGc_mrc': 0.0, 'WRDFAMI_mrc': 0.0,
            'WRDHYPnv': 0.0,
            'SMCAUSwn_1_binary': 0.0, 'SMCAUSwn_a_binary': 0.0,
            'SMCAUSwn_1p_binary': 0.0, 'SMCAUSwn_ap_binary': 0.0,
            'SMCAUSwn_1_path': 0.0, 'SMCAUSwn_a_path': 0.0,
            'SMCAUSwn_1p_path': 0.0, 'SMCAUSwn_ap_path': 0.0,
            'SMCAUSwn_1_lch': 0.0, 'SMCAUSwn_a_lch': 0.0,
            'SMCAUSwn_1p_lch': 0.0, 'SMCAUSwn_ap_lch': 0.0,
            'SMCAUSwn_1_wup': 0.0, 'SMCAUSwn_a_wup': 0.0,
            'SMCAUSwn_1p_wup': 0.0, 'SMCAUSwn_ap_wup': 0.0,
        }

    def _compute_referential_cohesion(self, sentence_embeddings: Dict[str, List[np.ndarray]]) -> Dict[str, float]:
        """Compute referential cohesion scores"""
        if not sentence_embeddings:
            return {
                'PCREF_1': 0.0, 'PCREF_a': 0.0, 
                'PCREF_1p': 0.0, 'PCREF_ap': 0.0
            }
            
        # Get all embeddings in document order
        all_embeddings = []
        for p_id, embeddings in sentence_embeddings.items():
            for embedding in embeddings:
                all_embeddings.append(embedding)

        # Document-level scores
        local_cosine = self.embeddings_helper.compute_local_cosine(all_embeddings)
        global_cosine = self.embeddings_helper.compute_global_cosine(all_embeddings)

        # Paragraph-level scores - only compute for paragraphs with multiple sentences
        local_scores = {}
        global_scores = {}
        for p_id, embeddings in sentence_embeddings.items():
            if len(embeddings) > 1:  # Skip single-sentence paragraphs for local
                local_scores[p_id] = self.embeddings_helper.compute_local_cosine(embeddings)
            if len(embeddings) > 0:  # Include all paragraphs for global
                global_scores[p_id] = self.embeddings_helper.compute_global_cosine(embeddings)

        # Mean of paragraph-level scores
        local_cosine_p = statistics.mean(list(local_scores.values())) if local_scores else 0.0
        global_cosine_p = statistics.mean(list(global_scores.values())) if global_scores else 0.0

        return {
            'PCREF_1': local_cosine,
            'PCREF_a': global_cosine,
            'PCREF_1p': local_cosine_p,
            'PCREF_ap': global_cosine_p
        }

    def _compute_deep_cohesion(self, doc: DocumentStructure) -> float:
        """Compute deep cohesion score based on causal patterns"""
        if doc.sentence_count == 0:
            return 0.0
            
        # Count patterns in all sentences
        total_patterns = sum(
            self.causal_helper.count_causal_patterns(sentence)
            for sentences in doc.sentences.values()
            for sentence in sentences
        )
                
        return total_patterns / doc.sentence_count

    def _compute_token_based_indices(self, doc: DocumentStructure, token_ids_by_sentence: Dict[str, list]) -> Dict[str, float]:
        """Compute all token-based indices in a single pass"""
        if doc.tokens.empty:
            return self._get_token_based_default_indices()
            
        # Create accumulators
        concreteness_scores, imageability_scores, familiarity_scores, hypernymy_scores = self._initialize_accumulators()
        semantic_scores = self._initialize_semantic_accumulators()
        wordnet_scores = self._initialize_wordnet_accumulators()
        
        # Process token-level lexical features (concreteness, imageability, hypernymy)
        if self.enabled_groups['concreteness'] or self.enabled_groups['wordnet']:
            self._process_lexical_features(
                doc.tokens, 
                concreteness_scores, 
                imageability_scores, 
                familiarity_scores,
                hypernymy_scores
            )
        
        # Process verb-level features (causality and wordnet relational features)
        if self.enabled_groups['causality'] or self.enabled_groups['wordnet']:
            verb_data = self.text_processor.filter_tokens_by_pos(
                doc, token_ids_by_sentence, pos_tags=['VERB']
            )
            all_verbs = self._process_verb_metrics(verb_data, semantic_scores, wordnet_scores)
        
        # Build final results
        return self._build_token_based_results(
            semantic_scores, 
            wordnet_scores,
            concreteness_scores, 
            imageability_scores, 
            familiarity_scores,
            hypernymy_scores
        )

    def _initialize_accumulators(self):
        """Initialize accumulators for various feature types"""
        concreteness_scores = {'megahr': [], 'mrc': []}
        imageability_scores = {'megahr': [], 'mrc': []}
        familiarity_scores = {'mrc': []}
        hypernymy_scores = []
        return concreteness_scores, imageability_scores, familiarity_scores, hypernymy_scores

    def _initialize_semantic_accumulators(self):
        """Initialize accumulators for semantic scores"""
        return {
            'local': [], 'global': [], 
            'para_local': [], 'para_global': []
        }

    def _initialize_wordnet_accumulators(self):
        """Initialize accumulators for WordNet scores"""
        return {
            metric: {'local': [], 'global': [], 'para_local': [], 'para_global': []}
            for metric in ['path', 'lch', 'wup', 'binary']
        }

    def _process_lexical_features(self, tokens_df, concreteness_scores, imageability_scores, 
                                familiarity_scores, hypernymy_scores):
        """Process lexical features for all tokens"""
        process_concreteness = self.enabled_groups['concreteness'] and (self.megahr_dict is not None or self.mrc_db is not None)
        process_wordnet = self.enabled_groups['wordnet']
        
        # Process in batches for memory efficiency
        batch_size = 1000
        for start_idx in range(0, len(tokens_df), batch_size):
            end_idx = min(start_idx + batch_size, len(tokens_df))
            batch = tokens_df.iloc[start_idx:end_idx]
            
            for _, row in batch.iterrows():
                token_text = row['token_text'].lower()
                token_pos = row['token_pos']
                
                # MegaHR scores (concreteness)
                if process_concreteness and self.megahr_dict:
                    self._process_megahr_scores(token_text, concreteness_scores, imageability_scores)
                
                # MRC scores (concreteness)
                if process_concreteness and self.mrc_db:
                    self._process_mrc_scores(token_text, token_pos, concreteness_scores, 
                                            imageability_scores, familiarity_scores)
                
                # Hypernymy (for nouns and verbs only)
                if process_wordnet and token_pos in ['VERB', 'NOUN']:
                    self._process_hypernymy(token_text, token_pos, hypernymy_scores)

    def _process_megahr_scores(self, token_text, concreteness_scores, imageability_scores):
        """Process MegaHR scores for a token"""
        megahr_record = self.megahr_dict.find_word(token_text)
        if megahr_record:
            concreteness_scores['megahr'].append(megahr_record.conc)
            imageability_scores['megahr'].append(megahr_record.imag)

    def _process_mrc_scores(self, token_text, token_pos, concreteness_scores, 
                            imageability_scores, familiarity_scores):
        """Process MRC scores for a token"""
        mrc_records = self.mrc_db.find_word(token_text, token_pos)
        for record in mrc_records:
            concreteness_scores['mrc'].append(record.conc)
            imageability_scores['mrc'].append(record.imag)
            familiarity_scores['mrc'].append(record.fami)

    def _process_hypernymy(self, token_text, token_pos, hypernymy_scores):
        """Process hypernymy for a token"""
        score = self.wordnet_helper.compute_hypernymy_score(token_text, token_pos)
        if score > 0:
            hypernymy_scores.append(score)

    def _process_verb_metrics(self, verb_data, semantic_scores, wordnet_scores):
        """Process metrics based on verbs"""
        process_causality = self.enabled_groups['causality']
        process_wordnet = self.enabled_groups['wordnet']
        all_verbs = []
        
        # Process paragraph-level metrics
        for p_id, sentences in verb_data.items():
            # Process local (consecutive) sentence scores
            if len(sentences) > 1:
                for i in range(len(sentences) - 1):
                    self._process_consecutive_sentences(
                        sentences[i], sentences[i+1], 
                        semantic_scores, wordnet_scores, 
                        process_causality, process_wordnet
                    )
            
            # Process paragraph-level global scores
            para_tokens = []
            for sent in sentences:
                para_tokens.extend(sent)
                
            if len(para_tokens) > 1:
                self._process_paragraph_tokens(
                    para_tokens, semantic_scores, wordnet_scores,
                    process_causality, process_wordnet
                )
            
            # Collect all verbs for document-level processing
            for sent in sentences:
                all_verbs.extend(sent)
        
        # Process document-level metrics
        if len(all_verbs) > 1:
            self._process_document_verbs(
                all_verbs, semantic_scores, wordnet_scores,
                process_causality, process_wordnet
            )
            
        return all_verbs

    def _process_consecutive_sentences(self, sent1, sent2, semantic_scores, wordnet_scores,
                                    process_causality, process_wordnet):
        """Process metrics for consecutive sentences"""
        # Semantic similarity (causality)
        if process_causality:
            self._compute_sentence_semantic_similarity(
                sent1, sent2, semantic_scores
            )
        
        # WordNet similarity
        if process_wordnet:
            self._compute_sentence_wordnet_similarity(
                sent1, sent2, wordnet_scores, 'para_local'
            )

    def _compute_sentence_semantic_similarity(self, sent1, sent2, semantic_scores):
        """Compute semantic similarity between sentences"""
        sent1_pairs = [(t['id'], t['embedding']) for t in sent1]
        sent2_pairs = [(t['id'], t['embedding']) for t in sent2]
        
        if sent1_pairs and sent2_pairs:
            token_embeddings = {id: emb for id, emb in sent1_pairs + sent2_pairs}
            
            all_pairs = list(itertools.product(
                [id for id, _ in sent1_pairs],
                [id for id, _ in sent2_pairs]
            ))
                
            similarities = self.embeddings_helper.compute_token_similarities(
                token_embeddings, all_pairs
            )
            if similarities:
                semantic_scores['para_local'].append(statistics.mean(similarities.values()))

    def _compute_sentence_wordnet_similarity(self, sent1, sent2, wordnet_scores, score_type):
        """Compute WordNet similarity between sentences"""
        pairs = list(itertools.product(sent1, sent2))
        sentence_scores = {metric: [] for metric in ['path', 'lch', 'wup', 'binary']}
        
        for t1, t2 in pairs:
            pair = ((t1['text'], t1['synsets']), (t2['text'], t2['synsets']))
            scores = self.wordnet_helper.compute_synset_pair_similarity(pair)
            for metric in scores:
                sentence_scores[metric].append(scores[metric])
        
        # Average scores for this pair of sentences
        for metric in sentence_scores:
            if sentence_scores[metric]:
                wordnet_scores[metric][score_type].append(
                    statistics.mean(sentence_scores[metric])
                )

    def _process_paragraph_tokens(self, para_tokens, semantic_scores, wordnet_scores,
                                process_causality, process_wordnet):
        """Process tokens at the paragraph level"""
        # Semantic cohesion
        if process_causality:
            embeddings = [v['embedding'] for v in para_tokens]
            score = self.embeddings_helper.compute_global_cosine(embeddings)
            if score > 0:
                semantic_scores['para_global'].append(score)
        
        # WordNet similarity within paragraph
        if process_wordnet:
            self._compute_token_group_wordnet_similarity(
                para_tokens, wordnet_scores, 'para_global'
            )

    def _compute_token_group_wordnet_similarity(self, tokens, wordnet_scores, score_type):
        """Compute WordNet similarity for a group of tokens"""
        group_scores = {metric: [] for metric in ['path', 'lch', 'wup', 'binary']}
        pairs = list(itertools.combinations(tokens, 2))
        
        for t1, t2 in pairs:
            pair = ((t1['text'], t1['synsets']), (t2['text'], t2['synsets']))
            scores = self.wordnet_helper.compute_synset_pair_similarity(pair)
            for metric in scores:
                group_scores[metric].append(scores[metric])
        
        # Average scores
        for metric in group_scores:
            if group_scores[metric]:
                wordnet_scores[metric][score_type].append(
                    statistics.mean(group_scores[metric])
                )

    def _process_document_verbs(self, all_verbs, semantic_scores, wordnet_scores,
                            process_causality, process_wordnet):
        """Process all verbs at the document level"""
        # Semantic cohesion
        if process_causality:
            embeddings = [v['embedding'] for v in all_verbs]
            semantic_scores['local'].append(self.embeddings_helper.compute_local_cosine(embeddings))
            semantic_scores['global'].append(self.embeddings_helper.compute_global_cosine(embeddings))
        
        # WordNet global similarity
        if process_wordnet:
            self._compute_token_group_wordnet_similarity(
                all_verbs, wordnet_scores, 'global'
            )
            
            # Local consecutive pairs at document level
            for i in range(len(all_verbs) - 1):
                t1, t2 = all_verbs[i], all_verbs[i + 1]
                pair = ((t1['text'], t1['synsets']), (t2['text'], t2['synsets']))
                scores = self.wordnet_helper.compute_synset_pair_similarity(pair)
                for metric in scores:
                    wordnet_scores[metric]['local'].append(scores[metric])

    def _build_token_based_results(self, semantic_scores, wordnet_scores,
                                concreteness_scores, imageability_scores, 
                                familiarity_scores, hypernymy_scores):
        """Build final results dictionary from all scores"""
        # Start with default values for all metrics
        results = self._get_token_based_default_indices()
        
        # Define metric groups and their computation functions
        metric_groups = {
            'causality': self._build_causality_metrics,
            'concreteness': self._build_concreteness_metrics,
            'wordnet': self._build_wordnet_metrics
        }
        
        # Compute metrics for each enabled group
        for group_name, build_fn in metric_groups.items():
            if self.enabled_groups[group_name]:
                results.update(build_fn(
                    semantic_scores, wordnet_scores,
                    concreteness_scores, imageability_scores,
                    familiarity_scores, hypernymy_scores
                ))
        
        return results

    def _build_causality_metrics(self, semantic_scores, *args):
        """Build causality metrics dictionary"""
        return {
            'SMCAUSe_1': statistics.mean(semantic_scores['local']) if semantic_scores['local'] else 0.0,
            'SMCAUSe_a': statistics.mean(semantic_scores['global']) if semantic_scores['global'] else 0.0,
            'SMCAUSe_1p': statistics.mean(semantic_scores['para_local']) if semantic_scores['para_local'] else 0.0,
            'SMCAUSe_ap': statistics.mean(semantic_scores['para_global']) if semantic_scores['para_global'] else 0.0,
        }

    def _build_concreteness_metrics(self, semantic_scores, wordnet_scores,
                                concreteness_scores, imageability_scores,
                                familiarity_scores, *args):
        """Build concreteness metrics dictionary"""
        return {
            'PCCNC_megahr': statistics.mean(concreteness_scores['megahr']) if concreteness_scores['megahr'] else 0.0,
            'WRDIMGc_megahr': statistics.mean(imageability_scores['megahr']) if imageability_scores['megahr'] else 0.0,
            'PCCNC_mrc': statistics.mean(concreteness_scores['mrc']) if concreteness_scores['mrc'] else 0.0,
            'WRDIMGc_mrc': statistics.mean(imageability_scores['mrc']) if imageability_scores['mrc'] else 0.0,
            'WRDFAMI_mrc': statistics.mean(familiarity_scores['mrc']) if familiarity_scores['mrc'] else 0.0,
        }

    def _build_wordnet_metrics(self, semantic_scores, wordnet_scores,
                            concreteness_scores, imageability_scores,
                            familiarity_scores, hypernymy_scores):
        """Build WordNet metrics dictionary"""
        # Begin with hypernymy score
        results = {
            'WRDHYPnv': statistics.mean(hypernymy_scores) if hypernymy_scores else 0.0
        }
        
        # Add detailed WordNet scores
        scope_map = {
            '1': 'local',
            'a': 'global',
            '1p': 'para_local',
            'ap': 'para_global'
        }
        
        for metric in wordnet_scores:
            for scope in ['1', 'a', '1p', 'ap']:
                key = f'SMCAUSwn_{scope}_{metric}'
                mapped_scope = scope_map[scope]
                scores = wordnet_scores[metric][mapped_scope]
                results[key] = statistics.mean(scores) if scores else 0.0
        
        return results
            
    def _get_token_based_default_indices(self):
        """Returns default values for token-based indices"""
        indices = {
            'SMCAUSe_1': 0.0, 'SMCAUSe_a': 0.0, 
            'SMCAUSe_1p': 0.0, 'SMCAUSe_ap': 0.0,
            'PCCNC_megahr': 0.0, 'WRDIMGc_megahr': 0.0,
            'PCCNC_mrc': 0.0, 'WRDIMGc_mrc': 0.0,
            'WRDFAMI_mrc': 0.0, 'WRDHYPnv': 0.0
        }
        
        # Add WordNet scores with default values
        for metric in ['path', 'lch', 'wup', 'binary']:
            for scope in ['1', 'a', '1p', 'ap']:
                indices[f'SMCAUSwn_{scope}_{metric}'] = 0.0
                
        return indices

class GisPyClassifier:
    """Classifier wrapper for GisPy metrics"""
    
    def __init__(self, 
                 model_name='en_core_web_trf',
                 megahr_path=None, 
                 mrc_path=None, 
                 use_corenlp=False, 
                 corenlp_memory='4G',
                 sentence_model='kamalkraj/BioSimCSE-BioLinkBERT-BASE',
                 enabled_metrics=None):
        """Initialize the GisPy classifier"""
        
        # Reference values for Wolfe normalization
        self.wolfe_mean_sd = {
            'SMCAUSlsa': {'mean': 0.097, 'sd': 0.04},
            'SMCAUSwn': {'mean': 0.553, 'sd': 0.096},
            'WRDIMGc': {'mean': 410.346, 'sd': 24.994},
            'WRDHYPnv': {'mean': 1.843, 'sd': 0.26}
        }
        
        # Core indices needed for GIS calculation
        self.required_indices = [
            'PCREF_ap',
            'PCDC',
            'SMCAUSe_1p',
            'SMCAUSwn_1p_binary',
            'PCCNC_megahr',
            'WRDIMGc_megahr',
            'WRDHYPnv'
        ]
        
        print("Initializing GisPy classifier...")
        self.gispy = GisPy(
            megahr_path=megahr_path,
            mrc_path=mrc_path,
            use_corenlp=use_corenlp,
            corenlp_memory=corenlp_memory,
            model_name=model_name,
            sentence_model=sentence_model,
            enabled_metrics=enabled_metrics
        )
        print("GisPy classifier initialized successfully")
    
    def predict_single(self, text):
        """Process a single text and return the raw metrics"""
        if not self.gispy:
            print("GisPy classifier not properly initialized")
            return {}
            
        return self.gispy.compute_indices(text)

    def predict_batch(self, texts, batch_size=32):
        """Process a batch of texts and return raw metrics for each"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict_single(text) for text in batch]
            results.extend(batch_results)
        return results
        
    def normalize_scores(self, scores_collection, use_wolfe: bool = False):
        """Normalize raw scores and calculate the General Inference Score (GIS)"""
        df = scores_collection

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(scores_collection)
        
        normalized_scores = []
        for _, row in df.iterrows():
            z_scores = {}
            
            for index in self.required_indices:
                if use_wolfe and index in self.wolfe_mean_sd:
                    ref = self.wolfe_mean_sd[index]
                    z_scores[f"z{index}"] = (row[index] - ref['mean']) / ref['sd']
                else:
                    values = df[index].values
                    mean = np.mean(values)
                    std = np.std(values)
                    z_scores[f"z{index}"] = ((row[index] - mean) / std).item() if std > 0 else 0
            
            gis = (
                z_scores['zPCREF_ap'] +
                z_scores['zPCDC'] +
                (z_scores['zSMCAUSe_1p'] - z_scores['zSMCAUSwn_1p_binary']) -
                z_scores['zPCCNC_megahr'] -
                z_scores['zWRDIMGc_megahr'] -
                z_scores['zWRDHYPnv']
            )
            
            normalized_scores.append({
                'raw_scores': row.to_dict(),
                'z_scores': z_scores,
                'gis': gis
            })
            
        return normalized_scores
    
if __name__ == "__main__":
    texts = [
        "Age-related memory decline involves multiple pathophysiological processes including reduced hippocampal neurogenesis, decreased synaptic plasticity, mitochondrial dysfunction, and altered neurotransmitter systems. Contributing factors include oxidative stress, inflammatory processes, and reduced BDNF expression. Prevention strategies target modifiable risk factors through cognitive stimulation, cardiovascular health maintenance, and optimization of metabolic parameters.",
        "As we age, our brain's ability to form and retrieve memories naturally decreases, like a computer that becomes slower over time. While some decline is normal, we can help maintain memory by keeping our brain active (through learning new skills and socializing), exercising regularly, eating a healthy diet, and controlling blood pressure. Think of it as maintaining a garden - regular care and the right conditions help keep it healthy longer.",
        "Post-operative delirium results from complex interactions between predisposing and precipitating factors affecting neurotransmitter balance and cerebral metabolism. Common triggers include medications (especially anticholinergics and benzodiazepines), pain, infection, electrolyte disturbances, and sleep disruption. Prevention strategies include multicomponent non-pharmacological interventions (HELP protocol), while management focuses on identifying and treating underlying causes, maintaining orientation, and judicious use of antipsychotics for severe agitation.",
        "What your father experienced is called delirium, a temporary state of confusion that's common in older adults after surgery. It's like their brain's normal functions get temporarily scrambled due to the stress of surgery, medications, being in an unfamiliar place, or other medical issues. This usually improves as the person recovers, especially when they're in a calm environment with familiar faces and normal day-night routines. The medical team will look for and treat any underlying causes, like infection or medication effects, that might be contributing to the confusion.",
    ]

    # Paths to required resources
    megahr_path = "/data/home/djbf/storage/bls/resources/datasets/megahr/megahr.en.sort.i.txt" 
    mrc_path = "/data/home/djbf/storage/bls/resources/datasets/mrc/mrc_psycholinguistic_database.csv"


    # Time the execution
    start = timeit.default_timer()

    # Initialize calculator
    classifier = GisPyClassifier(
        megahr_path=megahr_path,
        mrc_path=mrc_path,
        use_corenlp=False,
        sentence_model='sentence-transformers/all-mpnet-base-v2'
    )

    # Compute raw scores for all texts
    raw_scores = classifier.predict_batch(texts)
    
    # Normalize scores across the collection
    normalized_results = classifier.normalize_scores(raw_scores)
        
    # Print results
    for i, result in enumerate(normalized_results):
        print(f"\nText {i + 1}:")
        print(f"GIS Score: {result['gis']:.3f}")
        print("\nKey component z-scores:")
        for key in sorted(result['z_scores'].keys()):
            print(f"{key}: {result['z_scores'][key]:.3f}")
            

    stop = timeit.default_timer()

    print(f"\nExecution time: {stop - start:.3f} seconds")