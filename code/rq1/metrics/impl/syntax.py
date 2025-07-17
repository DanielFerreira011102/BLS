from collections import defaultdict
from tqdm import tqdm

import spacy

class SyntaxClassifier:
    """Classifier for calculating syntax-based complexity metrics"""
    
    def __init__(self, model_name='en_core_web_trf'):
        """Initialize the syntax metrics classifier with a spaCy model"""       
        spacy.prefer_gpu()  # Use GPU if available
 
        # Load spaCy model directly
        self.nlp = spacy.load(model_name, disable={'ner', 'lemmatizer'})
        
        if 'tagger' not in self.nlp.pipe_names:
            self.nlp.enable_pipe('tagger')
        
        if 'parser' not in self.nlp.pipe_names:
            self.nlp.enable_pipe('parser')
            
        # Content words
        self.content_pos = {'NOUN', 'ADJ', 'VERB', 'ADV'}
        # Function words
        self.function_pos = {'AUX', 'ADP', 'DET', 'CONJ', 'CCONJ', 'SCONJ', 'PART'}
        # Individual POS categories
        self.noun_pos = {'NOUN', 'PROPN'}  # Including proper nouns
        self.adjective_pos = {'ADJ'}       # Adjectives
        self.verb_pos = {'VERB'}           # Verbs (excluding auxiliaries)
        self.adverb_pos = {'ADV'}          # Adverbs
        self.sconj_pos = {'SCONJ'}         # Subordinating conjunctions
        self.cconj_pos = {'CCONJ'}         # Coordinating conjunctions
        self.aux_pos = {'AUX'}             # Auxiliary verbs

    def predict_single(self, text):
        """Calculate syntax metrics for a text."""
        doc = self.nlp(text)
        return self._process_syntax(doc)

    def predict_batch(self, texts, batch_size=32):
        """Calculate syntax metrics for multiple texts using spaCy's efficient batch processing."""
        results = []
        
        # Process in batches using spaCy's pipe for efficiency
        docs = self.nlp.pipe(texts, batch_size=batch_size)
        for doc in tqdm(docs, total=len(texts), desc="Processing syntax metrics"):
            results.append(self._process_syntax(doc))
                
        return results

    def _process_syntax(self, doc):
        """Calculate syntax metrics from a spaCy doc."""
        # Count parts of speech
        pos_counts = defaultdict(int)
        total_words = 0
        sentences = list(doc.sents)
        sentence_count = len(sentences)
        
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counts[token.pos_] += 1
                total_words += 1
        
        # Handle empty text
        if total_words == 0:
            return {
                'content_ratio': 0.0,
                'function_ratio': 0.0,
                'noun_ratio': 0.0,
                'adjective_ratio': 0.0,
                'verb_ratio': 0.0,
                'adverb_ratio': 0.0,
                'sconj_ratio': 0.0,
                'cconj_ratio': 0.0,
                'aux_ratio': 0.0,
                'avg_dependency_distance': 0.0,
                'passive_ratio': 0.0,
                'negation_density': 0.0,
                'modal_ratio': 0.0,
                'avg_np_length': 0.0,
                'lr_asymmetry': 0.0,
                'embedding_depth': 0.0
            }
        
        # Calculate ratios for POS categories
        content_words = sum(pos_counts[pos] for pos in self.content_pos)
        function_words = sum(pos_counts[pos] for pos in self.function_pos)
        noun_count = sum(pos_counts[pos] for pos in self.noun_pos)
        adjective_count = sum(pos_counts[pos] for pos in self.adjective_pos)
        verb_count = sum(pos_counts[pos] for pos in self.verb_pos)
        adverb_count = sum(pos_counts[pos] for pos in self.adverb_pos)
        sconj_count = sum(pos_counts[pos] for pos in self.sconj_pos)
        cconj_count = sum(pos_counts[pos] for pos in self.cconj_pos)
        aux_count = sum(pos_counts[pos] for pos in self.aux_pos)
        
        # Syntactic complexity - calculate dependency distances
        dep_distances = []
        for token in doc:
            if token.head is not token:  # Not the root
                distance = abs(token.i - token.head.i)
                dep_distances.append(distance)
        
        avg_dependency_distance = sum(dep_distances) / len(dep_distances) if dep_distances else 0
        
        # Passive voice constructions
        passive_count = sum(1 for token in doc if token.dep_ == "auxpass" or token.dep_ == "nsubjpass")
        passive_ratio = passive_count / sentence_count if sentence_count > 0 else 0
        
        # Negation density
        negation_markers = sum(1 for token in doc if token.dep_ == "neg")
        negation_density = negation_markers / sentence_count if sentence_count > 0 else 0
        
        # Modal verbs
        modal_count = sum(1 for token in doc if token.tag_ == "MD")
        modal_ratio = modal_count / total_words
        
        # Noun phrase complexity
        noun_phrases = list(doc.noun_chunks)
        avg_np_length = sum(len(list(np)) for np in noun_phrases) / len(noun_phrases) if noun_phrases else 0
        
        # Left-right asymmetry
        left_modifiers = sum(1 for token in doc if token.head.i > token.i)
        right_modifiers = sum(1 for token in doc if token.head.i < token.i)
        lr_asymmetry = abs(left_modifiers - right_modifiers) / (left_modifiers + right_modifiers) if (left_modifiers + right_modifiers) > 0 else 0
        
        # Embedding depth (maximum depth of dependency tree)
        embedding_depth = max(len(list(token.ancestors)) for token in doc) if doc else 0

        return {
            # Basic content/function word ratios
            'content_ratio': content_words / total_words,
            'function_ratio': function_words / total_words,
            
            # POS-specific ratios
            'noun_ratio': noun_count / total_words,
            'adjective_ratio': adjective_count / total_words,
            'verb_ratio': verb_count / total_words,
            'adverb_ratio': adverb_count / total_words,
            'sconj_ratio': sconj_count / total_words,
            'cconj_ratio': cconj_count / total_words,
            'aux_ratio': aux_count / total_words,
            
            # Syntactic complexity metrics
            'avg_dependency_distance': avg_dependency_distance,
            'passive_ratio': passive_ratio,
            'negation_density': negation_density,
            'modal_ratio': modal_ratio,
            'avg_np_length': avg_np_length,
            'lr_asymmetry': lr_asymmetry,
            'embedding_depth': embedding_depth
        }