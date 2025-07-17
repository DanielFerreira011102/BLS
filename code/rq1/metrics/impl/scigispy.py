import re
import itertools
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
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import fasttext
import fasttext.util

from rq1.metrics.utils.ic_corpus_stats import CorpusStats
from rq1.metrics.utils.tensor2attr import Tensor2Attr

####################################################################################################
# Data Classes and Base Components
####################################################################################################

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
        """Compute and cache similarity between token pairs"""
        # Convert tuples back to numpy arrays
        embedding1 = np.array(embedding1_tuple)
        embedding2 = np.array(embedding2_tuple)
        
        # Compute similarity
        return util.cos_sim(embedding1, embedding2).item()

    def compute_token_similarities(self, token_embeddings: Dict[int, np.ndarray], 
                                token_pairs: List[Tuple[int, int]]) -> Dict[str, float]:
        """Compute similarities between token pairs using cached method"""
        similarities = {}
        for token1_id, token2_id in token_pairs:
            # Sort IDs to ensure consistent cache keys
            if token1_id > token2_id:
                token1_id, token2_id = token2_id, token1_id
            
            # Convert embeddings to hashable tuples
            embedding1_tuple = tuple(token_embeddings[token1_id].tolist())
            embedding2_tuple = tuple(token_embeddings[token2_id].tolist())
            
            # Use the cached method
            similarity = self._compute_token_similarity(token1_id, token2_id, 
                                                     embedding1_tuple, embedding2_tuple)
            
            # Store with consistent key format
            pair_id = f"{token1_id}@{token2_id}"
            similarities[pair_id] = similarity
            
        return similarities


class HypernymyHelper:
    """Helper class for hypernymy metrics calculation"""
    def __init__(self):
        """Initialize with cache for hypernymy scores"""
        pass

    @lru_cache(maxsize=100000)
    def _get_synsets(self, word: str, pos_tag: str) -> Tuple[str, ...]:
        """Get and cache synsets for a word and POS tag"""
        synsets = wn.synsets(word, pos=pos_tag)
        # Convert to hashable tuple of synset names for caching
        return tuple(s.name() for s in synsets)

    def compute_wrdhyp_metrics(self, doc: DocumentStructure) -> Dict[str, float]:
        """Compute both normalized hypernym path length metrics efficiently"""
        all_norm_path_lengths = []
        all_partial_norm_path_lengths = []
        
        if doc.tokens.empty:
            return {
                'WRDHYP_partial_norm': 0.0,
                'WRDHYP_norm': 0.0
            }
        
        # Process tokens to compute hypernymy scores
        for _, row in doc.tokens.iterrows():
            if row['token_pos'] not in ['NOUN', 'VERB']:
                continue
            
            word = row['token_lemma'].lower()
            pos_tag = self._get_wordnet_pos(row['token_pos'])
            
            if not pos_tag:
                continue
                
            # Get synsets using cached method
            synset_names = self._get_synsets(word, pos_tag)
            if not synset_names:
                continue
                
            # Convert back to synset objects
            synsets = [wn.synset(name) for name in synset_names]

            # Group paths by root
            root_paths = {}
            for synset in synsets:
                paths = synset.hypernym_paths()
                for path in paths:
                    if not path:
                        continue
                    root = path[0].name()
                    if root not in root_paths:
                        root_paths[root] = []
                    root_paths[root].append(len(path))  # Store path length
            
            # First calculate partial_norm (using all roots)
            for root, lengths in root_paths.items():
                if lengths:
                    l1_norm = sum(lengths)
                    if l1_norm > 0:
                        normalized_lengths = [length/l1_norm for length in lengths]
                        all_partial_norm_path_lengths.extend(normalized_lengths)
            
            # Then for norm version, select representative root
            selected_root_paths = root_paths

            if len(root_paths) > 1:
                longest_path_length = 0
                longest_path_root = None
                
                for root, lengths in root_paths.items():
                    if lengths:
                        max_length = max(lengths)
                        if max_length > longest_path_length:
                            longest_path_length = max_length
                            longest_path_root = root
                
                selected_root_paths = {longest_path_root: root_paths[longest_path_root]} if longest_path_root else {}
            
            # Calculate normalized lengths for selected roots
            for root, lengths in selected_root_paths.items():
                if lengths:
                    l1_norm = sum(lengths)
                    if l1_norm > 0:
                        normalized_lengths = [length/l1_norm for length in lengths]
                        all_norm_path_lengths.extend(normalized_lengths)
        
        # Calculate final scores
        wrdhyp_partial_norm = (
            sum(all_partial_norm_path_lengths) / len(all_partial_norm_path_lengths)
            if all_partial_norm_path_lengths else 0.0
        )
        wrdhyp_norm = (
            sum(all_norm_path_lengths) / len(all_norm_path_lengths)
            if all_norm_path_lengths else 0.0
        )
        
        return {
            'WRDHYP_partial_norm': wrdhyp_partial_norm,
            'WRDHYP_norm': wrdhyp_norm
        }

    def _get_wordnet_pos(self, spacy_pos: str) -> Optional[str]:
        """Convert SpaCy POS tags to WordNet POS tags"""
        mapping = {
            'NOUN': wn.NOUN,
            'VERB': wn.VERB,
        }
        return mapping.get(spacy_pos)

class VerbOverlapHelper:
    """Helper class for verb overlap calculations with paragraph-level scope options"""
    
    def __init__(self, 
                 fasttext_path: str = None,
                 biowordvec_path: str = None):
        """Initialize with paths to specialized word embedding models"""
        self.fasttext_model = None
        self.biowordvec_model = None
        self.fasttext_path = fasttext_path
        self.biowordvec_path = biowordvec_path
        
    def _ensure_models_loaded(self):
        """Ensure models are loaded when needed"""
        if self.fasttext_path and self.fasttext_model is None:
            print("Loading FastText model...")
            self.fasttext_model = fasttext.load_model(self.fasttext_path)
        
        if self.biowordvec_path and self.biowordvec_model is None:
            print("Loading BioWordVec model...")
            self.biowordvec_model = fasttext.load_model(self.biowordvec_path)
    
    @lru_cache(maxsize=20000)
    def _get_word_vector(self, model_name: str, word: str) -> Tuple[float, ...]:
        """Get and cache word vector for a specific model"""
        # Check for fasttext model
        if model_name == 'fasttext' and self.fasttext_model:
            vector = self.fasttext_model.get_word_vector(word)
            return tuple(float(x) for x in vector)
        
        # Check for biowordvec model
        if model_name == 'biowordvec' and self.biowordvec_model:
            vector = self.biowordvec_model.get_word_vector(word)
            return tuple(float(x) for x in vector)
        
        # Return empty vector if no model is available
        return tuple([0.0] * 300)  # Default dimension
    
    def _compute_consecutive_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Compute cosine similarities between consecutive embeddings"""
        if len(embeddings) < 2:
            return []
            
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i+1])
            similarities.append(similarity)
            
        return similarities
    
    def _compute_all_pairs_similarities(self, embeddings: np.ndarray) -> List[float]:
        """Compute cosine similarities between all pairs of embeddings"""
        if len(embeddings) < 2:
            return []
            
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)
                
        return similarities
        
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)
    
    def compute_verb_overlap(self, doc: DocumentStructure) -> Dict[str, float]:
        """
        Compute verb overlap metrics with paragraph and document level scopes
        Returns dictionary with all 8 metrics:
        SMCAUSf_1, SMCAUSf_a, SMCAUSf_1p, SMCAUSf_ap,
        SMCAUSb_1, SMCAUSb_a, SMCAUSb_1p, SMCAUSb_ap
        """
        # Initialize results dictionary with zeros
        results = {
            'SMCAUSf_1': 0.0, 'SMCAUSf_a': 0.0, 
            'SMCAUSf_1p': 0.0, 'SMCAUSf_ap': 0.0,
            'SMCAUSb_1': 0.0, 'SMCAUSb_a': 0.0, 
            'SMCAUSb_1p': 0.0, 'SMCAUSb_ap': 0.0
        }
        
        # Early return if no models available
        if not self.fasttext_path and not self.biowordvec_path:
            return results
            
        # Load models if needed
        self._ensure_models_loaded()
        
        # Get all verb tokens
        verb_tokens = doc.tokens[doc.tokens['token_pos'] == 'VERB']
        if len(verb_tokens) < 2:
            return results
        
        # Configure which models to process
        models_to_process = []
        if self.fasttext_model:
            models_to_process.append(('fasttext', self.fasttext_model))
        if self.biowordvec_model:
            models_to_process.append(('biowordvec', self.biowordvec_model))
            
        # Return early if no models were loaded
        if not models_to_process:
            return results
            
        # Process each model
        for model_name, _ in models_to_process:
            prefix = 'SMCAUSf' if model_name == 'fasttext' else 'SMCAUSb'
            
            # DOCUMENT LEVEL
            # -------------------
            # Get all verb embeddings in document order
            all_verb_embeddings = []
            verb_tokens_by_p_id = {}
            
            for _, row in verb_tokens.iterrows():
                token_text = row['token_text'].lower()
                p_id = row['p_id']
                
                # Get vector using cached method
                vector_tuple = self._get_word_vector(model_name, token_text)
                embedding = np.array(vector_tuple)
                
                # Add to document list
                all_verb_embeddings.append(embedding)
                
                # Track by paragraph ID for paragraph-level metrics
                if p_id not in verb_tokens_by_p_id:
                    verb_tokens_by_p_id[p_id] = []
                verb_tokens_by_p_id[p_id].append(token_text)
            
            # Document-level consecutive pairs (1)
            consecutive_scores = self._compute_consecutive_similarities(np.array(all_verb_embeddings))
            results[f'{prefix}_1'] = statistics.mean(consecutive_scores) if consecutive_scores else 0.0
            
            # Document-level all pairs (a)
            all_pairs_scores = self._compute_all_pairs_similarities(np.array(all_verb_embeddings))
            results[f'{prefix}_a'] = statistics.mean(all_pairs_scores) if all_pairs_scores else 0.0
            
            # PARAGRAPH LEVEL
            # -------------------
            para_consecutive_scores = []
            para_all_pairs_scores = []
            
            for p_id, tokens in verb_tokens_by_p_id.items():
                if len(tokens) < 2:
                    continue
                    
                # Get embeddings for verbs in this paragraph
                p_embeddings = [np.array(self._get_word_vector(model_name, token)) for token in tokens]
                p_embeddings = np.array(p_embeddings)
                
                # Paragraph-level consecutive pairs
                p_consecutive = self._compute_consecutive_similarities(p_embeddings)
                if p_consecutive:
                    para_consecutive_scores.append(statistics.mean(p_consecutive))
                
                # Paragraph-level all pairs
                p_all_pairs = self._compute_all_pairs_similarities(p_embeddings)
                if p_all_pairs:
                    para_all_pairs_scores.append(statistics.mean(p_all_pairs))
            
            # Calculate paragraph-level metrics (averaged across paragraphs)
            results[f'{prefix}_1p'] = statistics.mean(para_consecutive_scores) if para_consecutive_scores else 0.0
            results[f'{prefix}_ap'] = statistics.mean(para_all_pairs_scores) if para_all_pairs_scores else 0.0
            
        return results
    
class CohesionHelper:
    """Helper class for semantic chunking analysis with multiple scoping options"""
    
    def __init__(self, model_name: str = "kamalkraj/BioSimCSE-BioLinkBERT-BASE", 
                 buffer_size: int = 1, 
                 breakpoint_percentile: int = 75):
        """Initialize with model and chunking parameters"""
        self.model = None  # Lazy loading
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.breakpoint_percentile = breakpoint_percentile
        
    def _ensure_model_loaded(self):
        """Ensure the model is loaded when needed"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
    
    @lru_cache(maxsize=10000)
    def _get_embedding(self, text: str) -> Tuple[float, ...]:
        """Get or retrieve cached embedding for a text segment"""
        self._ensure_model_loaded()
        embedding = self.model.encode(text, show_progress_bar=False)
        return tuple(float(x) for x in embedding)
        
    def get_embedding(self, text: str) -> np.ndarray:
        """Wrapper for the cached embedding method that returns numpy array"""
        embedding_tuple = self._get_embedding(text)
        return np.array(embedding_tuple)
    
    def compute_consecutive_similarities(self, embeddings: List[np.ndarray]) -> List[float]:
        """Compute similarities between consecutive sentence embeddings"""
        if len(embeddings) < 2:
            return []
            
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(similarity)
            
        return similarities
        
    def compute_all_pair_similarities(self, embeddings: List[np.ndarray]) -> List[float]:
        """Compute similarities between all pairs of sentence embeddings"""
        if len(embeddings) < 2:
            return []
            
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(similarity)
                
        return similarities
    
    def get_breakpoints(self, similarities: List[float]) -> List[int]:
        """Find breakpoints in a similarity list using percentile threshold"""
        if not similarities:
            return []
            
        # Smooth similarities using buffer_size
        smoothed_similarities = []
        for i in range(len(similarities)):
            start = max(0, i - self.buffer_size)
            end = min(len(similarities), i + self.buffer_size + 1)
            window = similarities[start:end]
            smoothed_similarity = np.mean(window)
            smoothed_similarities.append(smoothed_similarity)
        
        # Find breakpoints using percentile threshold
        breakpoint_threshold = np.percentile(smoothed_similarities, self.breakpoint_percentile)
        breakpoints = [i for i, s in enumerate(smoothed_similarities) if s < breakpoint_threshold]
        
        return breakpoints
    
    def chunk_text_consecutive(self, sentences: List[str]) -> List[int]:
        """Identify breakpoints between consecutive sentences"""
        if len(sentences) < 2:
            return []
            
        # Get embeddings for all sentences
        embeddings = [self.get_embedding(sent) for sent in sentences]
        
        # Compute similarities between consecutive sentences
        similarities = self.compute_consecutive_similarities(embeddings)
        
        # Find breakpoints
        return self.get_breakpoints(similarities)
    
    def chunk_text_all_pairs(self, sentences: List[str]) -> List[int]:
        """Identify breakpoints using all sentence pairs"""
        if len(sentences) < 2:
            return []
            
        # Get embeddings for all sentences
        embeddings = [self.get_embedding(sent) for sent in sentences]
        
        # Compute similarities between all pairs
        similarities = self.compute_all_pair_similarities(embeddings)
        
        # For all-pairs, create boundary scores for each sentence transition
        sentence_boundary_scores = [[] for _ in range(len(sentences)-1)]
        
        # Map pair similarities to sentence boundaries they affect
        pair_idx = 0
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if j - i == 1:  # Adjacent sentences
                    sentence_boundary_scores[i].append(similarities[pair_idx])
                pair_idx += 1
        
        # Determine breakpoints
        breakpoints = []
        if not similarities:
            return breakpoints
            
        threshold = np.percentile(similarities, self.breakpoint_percentile)
        
        # A boundary is a breakpoint if its average similarity is below threshold
        for i, scores in enumerate(sentence_boundary_scores):
            if scores and np.mean(scores) < threshold:
                breakpoints.append(i)
                
        return breakpoints
    
    def compute_chunk_density(self, sentences: List[str], use_all_pairs: bool = False) -> float:
        """Compute chunk density for a list of sentences"""
        if not sentences:
            return 0.0
            
        if len(sentences) == 1:
            return 1.0
            
        # Get breakpoints based on approach
        if use_all_pairs:
            breakpoints = self.chunk_text_all_pairs(sentences)
        else:
            breakpoints = self.chunk_text_consecutive(sentences)
            
        # Compute chunk density
        num_chunks = len(breakpoints) + 1
        return num_chunks / len(sentences)
    
    def compute_chunk_metrics(self, doc: DocumentStructure) -> Dict[str, float]:
        """Compute all four variants of chunk density metrics"""
        if not doc.sentences or doc.sentence_count == 0:
            return {
                'PCREF_chunk_1': 0.0,   # Consecutive pairs, document level
                'PCREF_chunk_a': 0.0,   # All pairs, document level
                'PCREF_chunk_1p': 0.0,  # Consecutive pairs, paragraph level
                'PCREF_chunk_ap': 0.0,  # All pairs, paragraph level
            }
        
        # Get all sentences from the document
        all_sentences = [s for sentences in doc.sentences.values() for s in sentences]
        
        # Document-level metrics
        doc_consecutive = self.compute_chunk_density(all_sentences, use_all_pairs=False)
        doc_all_pairs = self.compute_chunk_density(all_sentences, use_all_pairs=True)
        
        # Paragraph-level metrics
        para_consecutive_densities = []
        para_all_pairs_densities = []
        
        for _, sentences in doc.sentences.items():
            if sentences:
                para_consecutive_densities.append(
                    self.compute_chunk_density(sentences, use_all_pairs=False)
                )
                para_all_pairs_densities.append(
                    self.compute_chunk_density(sentences, use_all_pairs=True)
                )
        
        return {
            'PCREF_chunk_1': doc_consecutive,
            'PCREF_chunk_a': doc_all_pairs,
            'PCREF_chunk_1p': statistics.mean(para_consecutive_densities) if para_consecutive_densities else 0.0,
            'PCREF_chunk_ap': statistics.mean(para_all_pairs_densities) if para_all_pairs_densities else 0.0,
        }

class ICHelper:
    """Helper class for Information Content calculations"""
    
    def __init__(self, stats_dir: str = None):
        """Initialize with corpus statistics path"""
        self.stats = None
        self.stats_dir = stats_dir
        
    def _ensure_stats_loaded(self):
        """Ensure corpus statistics are loaded when needed"""
        if self.stats is None and self.stats_dir:
            self.stats = CorpusStats.load(self.stats_dir)
    
    def _get_term_ic(self, lemma: str) -> float:
        """Get and cache information content for a term"""
        self._ensure_stats_loaded()
        if self.stats is None:
            return 0.0
            
        return self.stats.get_term_ic(lemma)
    
    def compute_ic(self, doc: DocumentStructure) -> float:
        """Compute average Information Content for terms in document"""
        if doc.tokens.empty or not self.stats_dir:
            return 0.0
            
        self._ensure_stats_loaded()
        if self.stats is None:
            return 0.0
        
        ic_scores = []
        
        for _, row in doc.tokens.iterrows():
            # Only process nouns and verbs
            if row['token_pos'] not in ['NOUN', 'VERB']:
                continue
                    
            lemma = row['token_lemma'].lower()
            ic = self._get_term_ic(lemma)
            ic_scores.append(ic)
                
        # Return average IC
        return statistics.mean(ic_scores) if ic_scores else 0.0

####################################################################################################
# Core SciGisPy Implementation
####################################################################################################

class SciGisPy:
    """Main class for computing biomedical text inference scores"""
    
    def __init__(self, 
                 model_name='en_core_web_trf',
                 fasttext_path=None,
                 biowordvec_path=None,
                 stats_dir=None,
                 sentence_model='kamalkraj/BioSimCSE-BioLinkBERT-BASE',
                 window_size=1,
                 breakpoint_percentile=75,
                 enabled_metrics=None):
        """Initialize all helper components with configurable paths and settings"""
        
        # Configure which metrics to enable
        self._init_metrics(enabled_metrics)
        
        # Initialize text processor with model_name
        self.text_processor = TextProcessor(model_name)
        
        self.hypernymy_helper = HypernymyHelper()
        self.verb_overlap_helper = VerbOverlapHelper(
            fasttext_path=fasttext_path,
            biowordvec_path=biowordvec_path
        )
        self.cohesion_helper = CohesionHelper(
            model_name=sentence_model,
            buffer_size=window_size,
            breakpoint_percentile=breakpoint_percentile
        )        
        self.ic_helper = ICHelper(stats_dir=stats_dir)
        self.embeddings_helper = EmbeddingsHelper(model_name=sentence_model)
        
    def _init_metrics(self, enabled_metrics=None):
        """Configure which metrics to calculate."""
        # Default settings - enable all
        self.enabled_groups = {
            'hypernymy': True,    # Hypernymy metrics
            'verb_overlap': True, # Verb overlap metrics
            'cohesion': True,     # Semantic chunking and cohesion
            'ic': True,           # Information content
        }
        
        # Override with user-specified enabled metrics if provided
        if enabled_metrics is not None:
            for group in self.enabled_groups:
                if group in enabled_metrics:
                    self.enabled_groups[group] = enabled_metrics[group]
        
        # Log enabled metrics
        enabled_groups = [g for g, e in self.enabled_groups.items() if e]
        print(f"SciGisPy enabled metric groups: {enabled_groups}")
        
    def compute_indices(self, text: str) -> Dict[str, float]:
        """Compute all indices needed for SciGIS calculation"""
        if not text:
            return self._get_default_indices()
                
        # Process document
        doc = self.text_processor.process_document(text)
        
        # Initialize with default values for all indices
        indices = self._get_default_indices()
        
        # Early return for empty documents
        if doc.tokens.empty or doc.sentence_count == 0:
            return indices
            
        # Get token IDs by sentence for later use
        token_ids_by_sentence = self.text_processor.get_token_ids_by_sentence(doc)
        
        # Compute cohesion-based metrics
        if self.enabled_groups['cohesion']:
            indices.update(self._compute_cohesion_metrics(doc))
            
        # Check if any token-based metrics are needed
        token_based_metrics_needed = (
            self.enabled_groups['hypernymy'] or 
            self.enabled_groups['verb_overlap'] or 
            self.enabled_groups['ic']
        )
        
        # Compute token-based metrics in a single pass
        if token_based_metrics_needed:
            indices.update(self._compute_token_based_indices(doc, token_ids_by_sentence))
            
        return indices
    
    def _compute_cohesion_metrics(self, doc: DocumentStructure) -> Dict[str, float]:
        """Compute semantic cohesion metrics"""
        # Initialize with default values
        cohesion_indices = {
            'PCREF_chunk_1': 0.0,
            'PCREF_chunk_1p': 0.0,
            'PCREF_chunk_a': 0.0,
            'PCREF_chunk_ap': 0.0,
        }
        
        # Return defaults if document is empty
        if not doc.sentences or doc.sentence_count == 0:
            return cohesion_indices
        
        # Compute all semantic chunking variants
        return self.cohesion_helper.compute_chunk_metrics(doc)
    
    def _compute_token_based_indices(self, doc: DocumentStructure, 
                            token_ids_by_sentence: Dict[str, List[int]]) -> Dict[str, float]:
        """Compute all token-based indices in a single pass for efficiency"""
        # Create default result dictionary
        results = {
            'WRDHYP_partial_norm': 0.0, 'WRDHYP_norm': 0.0,
            'SMCAUSf_1': 0.0, 'SMCAUSf_a': 0.0, 
            'SMCAUSf_1p': 0.0, 'SMCAUSf_ap': 0.0,
            'SMCAUSb_1': 0.0, 'SMCAUSb_a': 0.0, 
            'SMCAUSb_1p': 0.0, 'SMCAUSb_ap': 0.0,
            'WRDIC': 0.0
        }
        
        # Early return for empty documents
        if doc.tokens.empty:
            return results
        
        # Compute hypernymy metrics if enabled
        if self.enabled_groups['hypernymy']:
            hypernymy_results = self.hypernymy_helper.compute_wrdhyp_metrics(doc)
            results.update(hypernymy_results)
            
        # Compute verb overlap metrics if enabled
        if self.enabled_groups['verb_overlap']:
            verb_overlap_results = self.verb_overlap_helper.compute_verb_overlap(doc)
            results.update(verb_overlap_results)
            
        # Compute information content if enabled
        if self.enabled_groups['ic']:
            results['WRDIC'] = self.ic_helper.compute_ic(doc)
            
        return results
        
    def _get_default_indices(self) -> Dict[str, float]:
        """Return default zero values for all indices"""
        return {
            'WRDHYP_partial_norm': 0.0,
            'WRDHYP_norm': 0.0,
            'SMCAUSf_1': 0.0, 'SMCAUSf_a': 0.0, 
            'SMCAUSf_1p': 0.0, 'SMCAUSf_ap': 0.0,
            'SMCAUSb_1': 0.0, 'SMCAUSb_a': 0.0, 
            'SMCAUSb_1p': 0.0, 'SMCAUSb_ap': 0.0,
            'PCREF_chunk_1': 0.0, 'PCREF_chunk_a': 0.0,
            'PCREF_chunk_1p': 0.0, 'PCREF_chunk_ap': 0.0,
            'WRDIC': 0.0,
        }

class SciGisPyClassifier:
    """Classifier wrapper for SciGisPy metrics"""
    
    def __init__(self, 
                 model_name='en_core_web_trf',
                 fasttext_path=None,
                 biowordvec_path=None,
                 stats_dir=None, 
                 sentence_model='kamalkraj/BioSimCSE-BioLinkBERT-BASE',
                 window_size=1,
                 breakpoint_percentile=75,
                 enabled_metrics=None):
        """Initialize the SciGisPy classifier"""
        
        # Reference values for biomedical text normalization
        self.reference_mean_sd = {
            'WRDHYP_norm': {'mean': 0.42, 'sd': 0.08},
            'SMCAUSf_a': {'mean': 0.65, 'sd': 0.12},
            'PCREF_chunk': {'mean': 1.8, 'sd': 0.4},
            'WRDIC': {'mean': 8.2, 'sd': 1.5}
        }
        
        # Core indices needed for SciGIS calculation
        self.required_indices = [
            'WRDHYP_norm',
            'SMCAUSf_a',
            'PCREF_chunk',
            'WRDIC',
        ]
        
        print("Initializing SciGisPy classifier...")
        self.scigispy = SciGisPy(
            fasttext_path=fasttext_path,
            biowordvec_path=biowordvec_path,
            stats_dir=stats_dir,
            model_name=model_name,
            sentence_model=sentence_model,
            window_size=window_size,
            breakpoint_percentile=breakpoint_percentile,
            enabled_metrics=enabled_metrics
        )
        print("SciGisPy classifier initialized successfully")
    
    def predict_single(self, text):
        """Process a single text and return the raw metrics"""
        if not self.scigispy:
            print("SciGisPy classifier not properly initialized")
            return {}
            
        return self.scigispy.compute_indices(text)

    def predict_batch(self, texts, batch_size=32):
        """Process a batch of texts and return raw metrics for each"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = [self.predict_single(text) for text in batch]
            results.extend(batch_results)
                
        return results
        
    def normalize_scores(self, scores_collection, use_reference=False):
        """Normalize raw scores and calculate the SciGIS Score"""
        df = scores_collection

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(scores_collection)
            
        normalized_scores = []
        for _, row in df.iterrows():
            z_scores = {}
            
            for index in self.required_indices:
                if use_reference and index in self.reference_mean_sd:
                    ref = self.reference_mean_sd[index]
                    z_scores[f"z{index}"] = (row[index] - ref['mean']) / ref['sd']
                else:
                    values = df[index].values
                    mean = np.mean(values)
                    std = np.std(values)
                    z_scores[f"z{index}"] = ((row[index] - mean) / std).item() if std > 0 else 0
            
            # Calculate SciGIS - the biomedical inference score
            scigis = (
                # + PCDC                        # More connectives improve text cohesion
                z_scores['zSMCAUSf_a'] +        # Higher verb overlap enhances coherence and simplicity
                -z_scores['zPCREF_chunk'] -     # Fewer semantic chunks indicate better coherence and simplicity
                -z_scores['zWRDIC']             # Lower information content (less specialized terminology) is preferred for simpler texts
                # + MSL                         # Shorter sentences are associated with greater simplicity
            )
            
            normalized_scores.append({
                'raw_scores': row.to_dict(),
                'z_scores': z_scores,
                'scigis': scigis
            })
            
        return normalized_scores


if __name__ == "__main__":
    texts = [
        "Recurrent nephrolithiasis often results from multiple factors including genetic predisposition affecting calcium metabolism, mutations in renal tubular transport proteins, and anatomical variations. Metabolic abnormalities such as hypercalciuria, hyperoxaluria, or hypocitraturia create supersaturated urine conditions. Environmental factors including dietary habits, fluid intake, and urinary pH additionally influence crystal formation and aggregation.",
        "Getting kidney stones repeatedly can be due to several factors working together. Some people inherit genes that affect how their bodies process minerals like calcium. Others may have specific dietary habits or medical conditions that make stones more likely to form. It's similar to how some people are more prone to getting cavities despite good dental care - a combination of genetics, diet, and other factors determines your risk.",
        "Key clinical manifestations include paroxysmal severe abdominal pain with periods of apparent wellness, accompanied by vomiting and currant jelly stools. Physical examination may reveal a sausage-shaped mass in the right upper quadrant. The condition represents telescoping of proximal bowel into distal segments, most commonly ileocolic, potentially leading to bowel obstruction, ischemia, and perforation if untreated.",
        "Parents should watch for episodes of sudden, severe belly pain where their toddler might pull their knees to their chest, followed by periods where they seem fine. Other warning signs include vomiting, unusual tiredness, and passing dark-red, jelly-like stools. These episodes typically occur every 15-20 minutes. If you notice these signs, seek immediate medical care.",
        "Two trials met the inclusion criteria. One compared 2% ketanserin ointment in polyethylene glycol (PEG) with PEG alone, used twice a day by 40 participants with arterial leg ulcers, for eight weeks or until healing, whichever was sooner. One compared topical application of blood-derived concentrated growth factor (CGF) with standard dressing (polyurethane film or foam); both applied weekly for six weeks by 61 participants with non-healing ulcers (venous, diabetic arterial, neuropathic, traumatic, or vasculitic). Both trials were small, reported results inadequately, and were of low methodological quality. Short follow-up times (six and eight weeks) meant it would be difficult to capture sufficient healing events to allow us to make comparisons between treatments. One trial demonstrated accelerated wound healing in the ketanserin group compared with the control group. In the trial that compared CGF with standard dressings, the number of participants with diabetic arterial ulcers were only reported in the CGF group (9/31), and the number of participants with diabetic arterial ulcers and their data were not reported separately for the standard dressing group. In the CGF group, 66.6% (6/9) of diabetic arterial ulcers showed more than a 50% decrease in ulcer size compared to 6.7% (2/30) of non-healing ulcers treated with standard dressing. We assessed this as very-low certainty evidence due to the small number of studies and arterial ulcer participants, inadequate reporting of methodology and data, and short follow-up period. Only one trial reported side effects (complications), stating that no participant experienced these during follow-up (six weeks, low-certainty evidence). It should also be noted that ketanserin is not licensed in all countries for use in humans. Neither study reported time to ulcer healing, patient satisfaction or quality of life. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers.",
        "We found two small studies that presented data for 49 participants with arterial leg ulcers (search conducted January 2019). The studies also included participants with other kinds of ulcers, and it is not clear what proportion of participants were diabetic. Neither study described the methods fully, both presented limited results for the arterial ulcer participants, and one study did not provide information on the number of participants with an arterial ulcer in the control group. The follow-up periods (six and eight weeks) were too short to measure healing. Therefore, the data that were available were incomplete and cannot be generalised to the greater population of people who suffer from arterial leg ulcers. One study randomised participants to either 2% ketanserin ointment in polyethylene glycol (PEG) or PEG alone, administered twice a day over eight weeks. This study reported increased wound healing in the ketanserin group, when compared with the control group. It should be noted that ketanserin is not licensed for use in humans in all countries. The second study randomised participants to either topically-applied growth factors isolated from the participant's own blood (concentrated growth factors (CGF)), or standard dressing; both applied weekly for six weeks. This study reported that 66.6% of CGF-treated diabetic arterial ulcers showed more than a 50% decrease in ulcer size, compared to 6.7% of non-healing ulcers treated with standard dressing. Only one study mentioned side effects, and reported that no participant experienced side effects during follow-up (six weeks). Neither of the two studies reported time to ulcer healing, patient satisfaction or quality of life measures. There is insufficient evidence to determine whether the choice of topical agent or dressing affects the healing of arterial leg ulcers. We downgraded the overall certainty of the available evidence to 'very low' and 'low', because the studies reported their methods poorly, there were only two studies and few participants with arterial disease, and because the studies were short and reported few results. This made it impossible to determine whether there was any real difference in the number of ulcers healed between the groups.",
        "Twenty-two eligible randomised trials were identified, of which 11 were crossover trials. The trials included 1099 women with 673 receiving an adrenergic drug (phenylpropanolamine in 11 trials, midodrine in two, norepinephrine in three, clenbuterol in another three, terbutaline in one, eskornade in one and Ro 115-1240 in one). No trials included men. The limited evidence suggested that an adrenergic agonist drug is better than placebo in reducing the number of pad changes and incontinence episodes, as well as improving subjective symptoms. In two small trials, the drugs also appeared to be better than pelvic floor muscle training, possibly reflecting relative acceptability of the treatments to women but perhaps due to differential withdrawal of women from the trial groups. There was not enough evidence to evaluate the use of higher compared to lower doses of adrenergic agonists nor the relative merits of an adrenergic agonist drug compared with oestrogen, whether used alone or in combination. Over a quarter of women reported adverse effects. There were similar numbers of adverse effects with adrenergics, placebo or alternative drug treatment. However, when these were due to recognised adrenergic stimulation (insomnia, restlessness and vasomotor stimulation) they were only severe enough to stop treatment in 4% of women. There was weak evidence to suggest that use of an adrenergic agonist was better than placebo treatment. There was not enough evidence to assess the effects of adrenergic agonists when compared to or combined with other treatments. Further larger trials are needed to identify when adrenergics may be useful. Patients using adrenergic agonists may suffer from minor side effects, which sometimes cause them to stop treatment. Rare but serious side effects, such as cardiac arrhythmias and hypertension, have been reported.",
        "This review of 22 trials involving 673 women and seven different adrenergic drugs found weak evidence that adrenergic agonists may help stress urinary incontinence. Side effects do occur but are usually minor. Rarely, more serious adverse effects such as high blood pressure can occur. More evidence is needed to compare adrenergic drugs with other drugs for stress incontinence and also with pelvic floor muscle exercises.",
    ]

    # Initialize classifier
    classifier = SciGisPyClassifier(
        fasttext_path="/data/home/djbf/storage/bls/resources/models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin",
        biowordvec_path="/data/home/djbf/storage/bls/resources/models/BioWordVec/BioWordVec_PubMed_MIMICIII_d200.bin",
        stats_dir="/data/home/djbf/storage/bls/resources/datasets/corpus_stats",
        model_name="en_core_web_trf"
    )
    
    for text in texts:
        indices = classifier.predict_single(text)
        print(indices)


