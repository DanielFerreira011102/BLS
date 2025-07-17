import pickle
import math
from typing import Set, Dict, List, Any, Optional

import numpy as np
import pandas as pd
import spacy
import torch

from quickumls import QuickUMLS

class QuickUmlsClassifier:
    """Classifier for UMLS (umls-2024AB-metathesaurus-full) concept extraction with length-normalized metrics and CHV familiarity scores."""
    
    def __init__(self, path_to_quickumls: str, lay_vocab_path: str, expert_vocab_path: str, 
                 chv_file_path: str, model_name: str = 'en_core_web_trf', 
                 threshold: float = 0.7, similarity_name: str = "jaccard", 
                 window: int = 5, overlapping_criteria: str = 'score'):
        """Initialize classifier with paths to resources and configuration."""
        spacy.prefer_gpu()  # Use GPU if available

        # Load spaCy model directly
        self.nlp = spacy.load(model_name, disable={'ner', 'parser'})
        
        # Initialize QuickUMLS
        self.matcher = QuickUMLS(
            quickumls_fp=path_to_quickumls, 
            threshold=threshold, 
            similarity_name=similarity_name, 
            window=window,
            overlapping_criteria=overlapping_criteria
        )
        self.matcher.nlp = self.nlp
        
        self.lay_vocab_path = lay_vocab_path
        self.expert_vocab_path = expert_vocab_path
        self.chv_file_path = chv_file_path

        # Load vocabulary sets and CHV data from file paths
        self.lay_set = self._load_vocab_set(lay_vocab_path)
        self.expert_set = self._load_vocab_set(expert_vocab_path)
        self.chv_data = self._load_chv_data(chv_file_path)

        # Create a dictionary for quick lookup of CHV scores by CUI
        self.chv_scores = self._build_chv_lookup()

    def _load_vocab_set(self, vocab_path: str) -> Set[str]:
        """Load a vocabulary set from a pickle file."""
        with open(vocab_path, 'rb') as f:
            vocab_set = pickle.load(f)
        
        if not isinstance(vocab_set, set):
            raise ValueError(f"Expected a set in {vocab_path}, got {type(vocab_set)}")
        
        return vocab_set


    def _load_chv_data(self, chv_path: str) -> pd.DataFrame:
        """Load CHV data from a CSV file."""
        chv_data = pd.read_csv(chv_path, low_memory=False, na_values=[""], keep_default_na=False)
        
        for col in ['Frequency Score', 'Context Score', 'CUI Score', 'Combo Score']:
            chv_data[col] = chv_data[col].replace(['-1', '-1.0', -1, -1.0, '\\N'], np.nan)
            chv_data[col] = pd.to_numeric(chv_data[col], errors='coerce')
        
        return chv_data

    def _build_chv_lookup(self) -> Dict[str, Dict[str, float]]:
        """Build a lookup dictionary for CHV scores by CUI."""
        chv_scores = {}
        if not self.chv_data.empty:
            for _, row in self.chv_data.iterrows():
                cui = row['CUI']
                chv_scores[cui] = {
                    'frequency_score': row['Frequency Score'],
                    'context_score': row['Context Score'],
                    'cui_score': row['CUI Score'],
                    'combo_score': row['Combo Score']
                }
        return chv_scores

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Extract UMLS concepts and calculate normalized metrics including CHV familiarity scores."""
        # Process the text once to get word count
        doc = self.nlp(text)
        word_count = sum(1 for token in doc if not token.is_punct and not token.is_space)
        
        # QuickUMLS processes raw text
        umls_terms = self.matcher.match(text, best_match=True, ignore_syntax=False)
        return self._process_umls_terms(umls_terms, word_count)

    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts sequentially."""
        return [self.predict_single(text) for text in texts]

    def _empty_result(self) -> Dict[str, Any]:
        """Return empty results when no terms are found or errors occur."""
        return {
            'term_matches': [],
            'term_density': 0,
            'lay_term_density': 0,
            'expert_term_density': 0,
            'core_expert_term_density': 0,
            'semantic_type_density': 0,
            'expert_to_lay_ratio': 0,
            'avg_term_length': 0,
            'unique_concept_density': 0,
            'term_repetition': 0,
            'semantic_diversity': 0,
            'top3_semtype_concentration': 0,
            'avg_matches_per_term': 0,
            'expert_term_ratio': 0,
            'lay_term_ratio': 0,
            'core_expert_term_ratio': 0,
            'avg_frequency_score': 0,
            'avg_context_score': 0,
            'avg_cui_score': 0,
            'avg_combo_score': 0
        }
    
    def _process_umls_terms(self, umls_terms: List[List[Dict]], word_count: int) -> Dict[str, Any]:
        """Process UMLS terms and calculate length-normalized metrics including CHV familiarity scores."""
        n_terms = len(umls_terms)
        if n_terms == 0:
            return self._empty_result()
        
        # Process matched terms
        term_metrics = []
        n_lay_terms = 0
        n_expert_terms = 0
        n_core_expert_terms = 0
        total_matches = 0
        all_semantic_types = set()
        term_lengths = []
        unique_cuis = set()
        semantic_type_counts = {}
        
        # Lists to store CHV scores for averaging
        frequency_scores = []
        context_scores = []
        cui_scores = []
        combo_scores = []
        
        for term_matches in umls_terms:
            total_matches += len(term_matches)
            
            # Track term length
            if term_matches:
                term_text = term_matches[0]['ngram']
                term_word_count = len(term_text.split())
                term_lengths.append(term_word_count)
                
                # Add CUIs to unique set
                for match in term_matches:
                    unique_cuis.add(match['cui'])
            
            # Check if terms are in vocabulary sets
            in_lay = any(match['cui'] in self.lay_set for match in term_matches)
            in_expert = any(match['cui'] in self.expert_set for match in term_matches)
            
            # Collect semantic types and count them
            for match in term_matches:
                for semtype in match['semtypes']:
                    semantic_type_counts[semtype] = semantic_type_counts.get(semtype, 0) + 1
                all_semantic_types.update(match['semtypes'])
            
            # Update counts
            if in_lay:
                n_lay_terms += 1
            if in_expert:
                n_expert_terms += 1
            if in_expert and not in_lay:
                n_core_expert_terms += 1
            
            # Store term info
            term_metrics.append({
                'matches': term_matches,
                'in_lay_set': in_lay,
                'in_expert_set': in_expert
            })

            # Extract CHV scores for terms that are in CHV (lay terms)
            if in_lay:
                for match in term_matches:
                    cui = match['cui']
                    if cui in self.chv_scores:
                        scores = self.chv_scores[cui]
                        frequency_scores.append(scores['frequency_score'])
                        context_scores.append(scores['context_score'])
                        cui_scores.append(scores['cui_score'])
                        combo_scores.append(scores['combo_score'])

        # Normalize metrics to per 100 words
        normalization_factor = 100 / word_count if word_count > 0 else 0
        
        # Calculate normalized metrics
        term_density = n_terms * normalization_factor
        lay_term_density = n_lay_terms * normalization_factor
        expert_term_density = n_expert_terms * normalization_factor
        core_expert_term_density = n_core_expert_terms * normalization_factor
        semantic_type_density = len(all_semantic_types) * normalization_factor
        
        # Calculate term complexity metrics
        avg_term_length = sum(term_lengths) / len(term_lengths) if term_lengths else 0
        unique_concept_density = len(unique_cuis) * normalization_factor
        
        # Calculate semantic type diversity (Shannon entropy)
        total_semtypes = sum(semantic_type_counts.values())
        semtype_entropy = -sum((count/total_semtypes) * math.log2(count/total_semtypes) 
                            for count in semantic_type_counts.values()) if total_semtypes > 0 else 0

        # Calculate semantic domain concentration
        sorted_semtypes = sorted(semantic_type_counts.items(), key=lambda x: x[1], reverse=True)
        top3_semtypes = sorted_semtypes[:3] if len(sorted_semtypes) >= 3 else sorted_semtypes
        top3_concentration = sum(count for _, count in top3_semtypes) / total_semtypes if total_semtypes > 0 else 0
        
        # Calculate other metrics
        expert_to_lay_ratio = n_expert_terms / n_lay_terms if n_lay_terms > 0 else (float('inf') if n_expert_terms > 0 else 0)
        avg_matches_per_term = total_matches / n_terms if n_terms > 0 else 0
        term_repetition = n_terms / len(unique_cuis) if len(unique_cuis) > 0 else 0
        expert_term_ratio = n_expert_terms / n_terms if n_terms > 0 else 0
        lay_term_ratio = n_lay_terms / n_terms if n_terms > 0 else 0
        core_expert_term_ratio = n_core_expert_terms / n_terms if n_terms > 0 else 0

        # Calculate average CHV familiarity scores
        avg_frequency_score = sum(frequency_scores) / len(frequency_scores) if frequency_scores else 0
        avg_context_score = sum(context_scores) / len(context_scores) if context_scores else 0
        avg_cui_score = sum(cui_scores) / len(cui_scores) if cui_scores else 0
        avg_combo_score = sum(combo_scores) / len(combo_scores) if combo_scores else 0

        return {
            'term_density': term_density,
            'lay_term_density': lay_term_density,
            'expert_term_density': expert_term_density,
            'core_expert_term_density': core_expert_term_density,
            'semantic_type_density': semantic_type_density,
            'expert_to_lay_ratio': expert_to_lay_ratio,
            'avg_term_length': avg_term_length,
            'unique_concept_density': unique_concept_density,
            'term_repetition': term_repetition,
            'semantic_diversity': semtype_entropy,
            'top3_semtype_concentration': top3_concentration,
            'avg_matches_per_term': avg_matches_per_term,
            'expert_term_ratio': expert_term_ratio,
            'lay_term_ratio': lay_term_ratio,
            'core_expert_term_ratio': core_expert_term_ratio,
            'avg_frequency_score': avg_frequency_score,
            'avg_context_score': avg_context_score,
            'avg_cui_score': avg_cui_score,
            'avg_combo_score': avg_combo_score
        }