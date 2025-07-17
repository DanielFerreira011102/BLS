"""
Medical jargon identification model based on the MEDREADME paper (Jiang & Xu, 2024).
Implements the 7-category fine-grained complex span identification approach from the paper.
https://arxiv.org/abs/2405.02144

---- MEDREADME Paper Performance Metrics (Table 8) ----
Model performance on complex span identification task:

Token-Level Micro F1 Scores:
                      Binary    3-Class   7-Category
BERT-large:           86.1      80.9      67.9
RoBERTa-large:        86.8      82.3      68.6
BioBERT-large:        85.3      80.7      67.0
PubMedBERT-large:     85.7      82.3      68.3

Entity-Level Micro F1 Scores:
                      Binary    3-Class   7-Category
BERT-large:           78.5      74.1      43.9
RoBERTa-large:        80.2      75.9      67.9
BioBERT-large:        78.4      72.6      64.9
PubMedBERT-large:     79.0      75.2      66.5

RoBERTa-large achieved the best performance in both token-level and entity-level evaluation,
which is why it's used as the default model in this implementation.
"""

import os
import logging
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any, Set

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support, f1_score
import matplotlib.pyplot as plt
from torchcrf import CRF

from utils.helpers import setup_logging, save_json, load_json

logger = setup_logging(logging.DEBUG)


################################################################################
# MEDICAL JARGON CLASSIFIER
#
# Implementation of a classifier for medical jargon identification that uses
# the MedicalJargonIdentifier model to identify jargon spans and calculate
# various linguistic metrics.
################################################################################

class MedicalJargonClassifier:
    """Classifier for medical jargon identification with linguistic metrics computation."""
    
    # Jargon category mappings
    GOOGLE_EASY = 'medical-jargon-google-easy'
    GOOGLE_HARD = 'medical-jargon-google-hard'
    MEDICAL_NAME = 'medical-name-entity'
    GENERAL_COMPLEX = 'general-complex'
    GENERAL_MULTISENSE = 'general-medical-multisense'
    ABBR_MEDICAL = 'abbr-medical'
    ABBR_GENERAL = 'abbr-general'
    
    # Category lists for classification and metrics
    ALL_CATEGORIES = [GOOGLE_EASY, GOOGLE_HARD, MEDICAL_NAME, GENERAL_COMPLEX,
                      GENERAL_MULTISENSE, ABBR_MEDICAL, ABBR_GENERAL]
    
    # High-level category mappings
    MEDICAL_CATEGORIES = [GOOGLE_EASY, GOOGLE_HARD, MEDICAL_NAME]
    GENERAL_CATEGORIES = [GENERAL_COMPLEX, GENERAL_MULTISENSE]
    ABBREVIATION_CATEGORIES = [ABBR_MEDICAL, ABBR_GENERAL]
    
    # Difficulty mappings
    EASY_CATEGORIES = [GOOGLE_EASY, ABBR_GENERAL]
    HARD_CATEGORIES = [GOOGLE_HARD, MEDICAL_NAME, GENERAL_COMPLEX, GENERAL_MULTISENSE, ABBR_MEDICAL]
    
    def __init__(self, jargon_model=None, pretrained_model: str = 'roberta-large', 
                 model_path: Optional[str] = None, device: Optional[str] = None,
                 spacy_model: str = 'en_core_web_trf'):
        """Initialize the medical jargon classifier with a jargon identification model."""
        self.jargon_model = jargon_model
        self.pretrained_model = pretrained_model
        self.model_path = model_path
        self.device = device
        self.spacy_model = spacy_model
        self.nlp = None
    
    def _load_spacy(self):
        """Lazy-load spaCy model for word counting."""
        if self.nlp is not None:
            return

        import spacy
        spacy.prefer_gpu()  # Use GPU if available
        logger.info(f'Loading spaCy model {self.spacy_model}')
        self.nlp = spacy.load(self.spacy_model, disable=['ner', 'tagger', 'lemmatizer', 'textcat'])
    
    def _load_jargon_model(self):
        """Lazy-load the jargon model if not already loaded."""
        if self.jargon_model is not None:
            return
            
        logger.info(f'Initializing jargon model with {self.pretrained_model}')
        self.jargon_model = MedicalJargonIdentifier(
            pretrained_model=self.pretrained_model, 
            device=self.device
        )
        
        if not self.model_path:
            return
            
        logger.info(f'Loading model weights from {self.model_path}')
        self.jargon_model.load_model(self.model_path)
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Extract jargon spans and calculate metrics for a single text."""
        if not text or not text.strip():
            logger.warning('Empty text provided for prediction')
            return self._empty_result()
        
        # Reuse predict_batch logic for a single text
        results = self.predict_batch([text])
        return results[0]
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Process multiple texts with true batching to calculate jargon metrics."""
        if not texts:
            return []
                
        self._load_jargon_model()
        self._load_spacy()
        
        # Get jargon spans for all texts
        logger.info(f"Predicting jargon spans for {len(texts)} texts...")
        jargon_results = self.jargon_model.predict_batch(texts, batch_size=batch_size)
        
        # Calculate metrics for each text using spaCy's batch processing
        logger.info(f"Calculating jargon metrics...")
        results = []
        
        # Process documents as they come through the pipeline
        docs = self.nlp.pipe(texts, batch_size=batch_size)
        
        for i, doc in enumerate(tqdm(docs, total=len(texts), desc="Calculating jargon metrics")):
            # Get the corresponding jargon result
            jargon_spans = jargon_results[i]['jargon_spans']
            
            # Calculate metrics
            word_count = sum(1 for token in doc if not token.is_punct and not token.is_space)
            char_count = len(doc.text)
            
            metrics = self._calculate_metrics(jargon_spans, word_count, char_count, doc)
            results.append(metrics)
        
        return results
    
    def _calculate_metrics(self, jargon_spans: List[Dict[str, Any]], 
                          word_count: int, char_count: int, doc) -> Dict[str, Any]:
        """Calculate jargon metrics based on identified spans."""
        if not jargon_spans:
            return self._empty_result()
        
        # Extract basic counts
        basic_counts = self._count_jargon_instances(jargon_spans, doc)
        
        # Calculate category metrics
        category_metrics = self._calculate_category_metrics(
            basic_counts, word_count, char_count)
        
        # Calculate sentence-level metrics
        sentence_metrics = self._calculate_sentence_metrics(jargon_spans, doc)
        
        # Calculate additional linguistic metrics
        linguistic_metrics = self._calculate_linguistic_metrics(
            jargon_spans, basic_counts, doc)
        
        # Combine all metrics
        return {
            **basic_counts['public_counts'],
            **category_metrics,
            **sentence_metrics,
            **linguistic_metrics
        }
    
    def _count_jargon_instances(self, jargon_spans: List[Dict[str, Any]], doc) -> Dict[str, Any]:
        """Count jargon spans and categorize them using the already processed spaCy doc."""
        # Count spans by category
        n_spans = len(jargon_spans)
        unique_terms = set(span['text'].lower() for span in jargon_spans)
        n_unique_terms = len(unique_terms)
        
        # Initialize counters
        high_level_counts = {'medical': 0, 'general': 0, 'abbreviation': 0}
        difficulty_counts = {'easy': 0, 'hard': 0}
        subcategory_counts = {}
        
        # Track span information
        span_lengths = []
        span_chars = []
        total_jargon_chars = 0
        total_span_words = 0
        
        # Process each span
        for span in jargon_spans:
            category = self._get_category(span['type'])
            
            # Count by subcategory
            subcategory_counts[category] = subcategory_counts.get(category, 0) + 1
            
            # Count by high-level category
            if category in self.MEDICAL_CATEGORIES:
                high_level_counts['medical'] += 1
            elif category in self.GENERAL_CATEGORIES:
                high_level_counts['general'] += 1
            elif category in self.ABBREVIATION_CATEGORIES:
                high_level_counts['abbreviation'] += 1
            
            # Count by difficulty level
            if category in self.EASY_CATEGORIES:
                difficulty_counts['easy'] += 1
            elif category in self.HARD_CATEGORIES:
                difficulty_counts['hard'] += 1
            
            # Use the character span positions to count words in the original doc
            span_start = span['start']
            span_end = span['end']
            span_text = span['text']
            
            # Count tokens in this character span using the existing spaCy doc
            doc_span = doc.char_span(span_start, span_end, alignment_mode="expand")

            span_word_count = sum(1 for token in doc_span if not token.is_punct and not token.is_space)            

            span_lengths.append(span_word_count)
            total_span_words += span_word_count
            span_chars.append(len(span_text))
            total_jargon_chars += len(span_text)
        
        return {
            # Public counts (included directly in the final metrics)
            'public_counts': {
                'jargon_count': n_spans,
            },
            # Private counts (used internally for calculations)
            'n_unique_terms': n_unique_terms,
            'high_level_counts': high_level_counts,
            'difficulty_counts': difficulty_counts,
            'subcategory_counts': subcategory_counts,
            'span_lengths': span_lengths,
            'span_chars': span_chars,
            'total_span_words': total_span_words,
            'total_jargon_chars': total_jargon_chars
        }
    
    def _calculate_category_metrics(self, counts: Dict[str, Any], 
                                   word_count: int, char_count: int) -> Dict[str, Any]:
        """Calculate category-related metrics with normalization."""
        # Extract counts
        n_spans = counts['public_counts']['jargon_count']
        n_unique_terms = counts['n_unique_terms']
        high_level_counts = counts['high_level_counts']
        difficulty_counts = counts['difficulty_counts']
        subcategory_counts = counts['subcategory_counts']
        total_span_words = counts['total_span_words']
        total_jargon_chars = counts['total_jargon_chars']
        
        # Normalization factor (per 100 words)
        normalization_factor = 100 / word_count if word_count else 0
        
        # Ratios
        easy_ratio = difficulty_counts['easy'] / n_spans if n_spans else 0
        hard_ratio = difficulty_counts['hard'] / n_spans if n_spans else 0
        hard_to_easy_ratio = difficulty_counts['hard'] / difficulty_counts['easy'] if difficulty_counts['easy'] else float('inf') if difficulty_counts['hard'] else 0
        medical_to_general_ratio = high_level_counts['medical'] / high_level_counts['general'] if high_level_counts['general'] else float('inf') if high_level_counts['medical'] else 0
        abbreviation_ratio = high_level_counts['abbreviation'] / n_spans if n_spans else 0
        
        # Calculate individual category density
        category_metrics = {f"{cat.replace('-', '_')}_density": 
                           subcategory_counts.get(cat, 0) * normalization_factor 
                           for cat in self.ALL_CATEGORIES}
        
        return {
            # Density metrics
            'jargon_density': n_spans * normalization_factor,
            'unique_jargon_density': n_unique_terms * normalization_factor,
            'jargon_span_word_density': total_span_words * normalization_factor,
            'jargon_char_ratio': total_jargon_chars / char_count if char_count else 0,
            
            # High-level category density
            'medical_density': high_level_counts['medical'] * normalization_factor,
            'general_density': high_level_counts['general'] * normalization_factor,
            'abbreviation_density': high_level_counts['abbreviation'] * normalization_factor,
            
            # Individual category density
            **category_metrics,
            
            # Difficulty metrics
            'easy_density': difficulty_counts['easy'] * normalization_factor,
            'hard_density': difficulty_counts['hard'] * normalization_factor,
            'easy_ratio': easy_ratio,
            'hard_ratio': hard_ratio,
            'hard_to_easy_ratio': hard_to_easy_ratio,
            
            # Semantic metrics
            'medical_to_general_ratio': medical_to_general_ratio,
            'abbreviation_ratio': abbreviation_ratio,
        }
    
    def _calculate_sentence_metrics(self, jargon_spans: List[Dict[str, Any]], 
                                   doc) -> Dict[str, Any]:
        """Calculate sentence-level jargon metrics."""
        # Get sentences from the pre-processed doc
        sentences = list(doc.sents)
        
        # Binary jargon presence
        binary_jargon_presence = 1 if jargon_spans else 0
        
        # Calculate sentence-level jargon density
        sentence_jargon_counts = []
        for sent in sentences:
            sent_start = sent.start_char
            sent_end = sent.end_char
            sent_jargon_spans = [span for span in jargon_spans 
                                if span['start'] >= sent_start and span['start'] < sent_end]
            sentence_jargon_counts.append(len(sent_jargon_spans))
        
        avg_sentence_jargon_density = sum(sentence_jargon_counts) / len(sentences) if sentences else 0
        
        return {
            'binary_jargon_presence': binary_jargon_presence,
            'avg_sentence_jargon_density': avg_sentence_jargon_density,
        }
    
    def _calculate_linguistic_metrics(self, jargon_spans: List[Dict[str, Any]], 
                                     counts: Dict[str, Any], doc) -> Dict[str, Any]:
        """Calculate linguistic complexity metrics for jargon spans."""
        # Extract counts
        n_spans = counts['public_counts']['jargon_count']
        n_unique_terms = counts['n_unique_terms']
        span_lengths = counts['span_lengths']
        span_chars = counts['span_chars']
        subcategory_counts = counts['subcategory_counts']
        
        # Jargon type diversity (Shannon entropy)
        jargon_diversity = 0
        if subcategory_counts:
            total_spans = sum(subcategory_counts.values())
            category_proportions = [count/total_spans for count in subcategory_counts.values()]
            # Ensure non-negative value due to floating-point precision
            jargon_diversity = max(0, -sum(p * np.log2(p) for p in category_proportions if p > 0))
        
        # Calculate jargon clustering index
        # Higher value = jargon terms appear closer together in the text
        jargon_clustering_index = 0
        
        # Skip clustering calculation if there aren't enough spans
        if len(jargon_spans) <= 1:
            jargon_clustering_index = 0
        else:
            # Find which token each jargon span starts at
            token_positions = []
            
            for span in jargon_spans:
                span_start_char = span['start']
                
                # Find which token contains this character position
                token_position = None
                for token in doc:
                    # Check if this token contains the span's start position
                    token_start = token.idx
                    token_end = token.idx + len(token.text)
                    
                    if token_start <= span_start_char < token_end:
                        token_position = token.i  # Store the token index
                        break
                
                # If we found a valid token, add it to our list
                if token_position is not None:
                    token_positions.append(token_position)
            
            # We need at least two token positions to calculate distances
            if len(token_positions) > 1:
                # Sort positions and calculate distances between consecutive jargon terms
                token_positions.sort()
                distances = [token_positions[i+1] - token_positions[i] 
                            for i in range(len(token_positions)-1)]
                
                # Calculate average distance between jargon terms
                avg_distance = sum(distances) / len(distances)
                
                # Convert to clustering index: higher value = more clustered
                jargon_clustering_index = 1.0 / (1.0 + avg_distance)
        
        return {
            # Complexity metrics
            'avg_jargon_word_length': sum(span_lengths) / len(span_lengths) if span_lengths else 0,
            'avg_jargon_char_length': sum(span_chars) / len(span_chars) if span_chars else 0,
            'jargon_repetition': n_spans / n_unique_terms if n_unique_terms else 0,
            'jargon_diversity': jargon_diversity,
            'jargon_clustering_index': jargon_clustering_index,
        }
    
    def _get_category(self, span_type: str) -> str:
        """Extract clean category name from span type without B-/I- prefix."""
        return span_type[2:] if span_type.startswith(('B-', 'I-')) else span_type
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty metrics when no jargon is detected."""
        category_metrics = {f"{cat.replace('-', '_')}_density": 0 for cat in self.ALL_CATEGORIES}
        return {
            # Basic counts
            'jargon_count': 0,
            
            # Density metrics
            'jargon_density': 0,
            'unique_jargon_density': 0,
            'jargon_span_word_density': 0,
            'jargon_char_ratio': 0,
            
            # High-level category densities
            'medical_density': 0,
            'general_density': 0,
            'abbreviation_density': 0,
            
            # Individual category densities
            **category_metrics,
            
            # Difficulty metrics
            'easy_density': 0,
            'hard_density': 0,
            'easy_ratio': 0,
            'hard_ratio': 0,
            'hard_to_easy_ratio': 0,
            
            # Complexity metrics
            'avg_jargon_word_length': 0,
            'avg_jargon_char_length': 0,
            'jargon_repetition': 0,
            'jargon_diversity': 0,
            'binary_jargon_presence': 0,
            'avg_sentence_jargon_density': 0,
            'jargon_clustering_index': 0,
            
            # Semantic metrics
            'medical_to_general_ratio': 0,
            'abbreviation_ratio': 0,
        }
    

################################################################################
# MEDICAL JARGON IDENTIFIER COMPONENTS
#
# Implementation of classes for fine-grained medical jargon identification
# based on the MEDREADME paper (Jiang & Xu, 2024). This includes the datasets,
# models, and utilities needed for training and inference.
################################################################################

class JargonDataset(Dataset):
    """Dataset for fine-grained medical jargon identification."""
    
    def __init__(self, examples: List[Dict], tokenizer, label_map: Dict[str, int], 
                 is_training: bool = True, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.id_to_label = {v: k for k, v in label_map.items()}
        self.is_training = is_training
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        tokens = example["tokens"]
        labels = example["labels"] if self.is_training else None
        
        # Tokenize for the model using transformer tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension for Dataset format
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if labels is None:
            return item
        
        # Add labels for training - handle subword tokenization properly
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens get "O" label for CRF
                label_ids.append(self.label_map["O"])
                continue
                
            if word_idx != previous_word_idx:
                # First token of a word gets the original label
                label_ids.append(labels[word_idx])
            else:
                # For continuation tokens, convert B- to I- if needed
                label = labels[word_idx]
                label_str = self.id_to_label.get(label, "O")
                
                if label_str.startswith("B-"):
                    # Convert B- to I- for continuation tokens
                    i_tag = "I-" + label_str[2:]
                    i_label = self.label_map.get(i_tag, label)
                    label_ids.append(i_label)
                else:
                    # Keep I- and O tags as is
                    label_ids.append(label)
            
            previous_word_idx = word_idx
        
        item["labels"] = torch.tensor(label_ids)
        # Use attention mask as label mask for CRF
        item["label_mask"] = item["attention_mask"].clone()
        
        return item


class CRFTokenClassificationModel(nn.Module):
    """Token classification model with a CRF layer on top."""
    
    def __init__(self, transformer_model, num_labels: int, dropout_rate: float = 0.1):
        super(CRFTokenClassificationModel, self).__init__()
        
        # Extract the base transformer model (without classification head)
        self.transformer = transformer_model.base_model
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification layer
        hidden_size = self.transformer.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)
        
        # CRF layer for improved sequence labeling
        self.crf = CRF(num_labels, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Apply dropout to last hidden state
        sequence_output = self.dropout(outputs.last_hidden_state)
        
        # Get logits from classification layer
        logits = self.classifier(sequence_output)
        
        # If no labels provided, return logits only (inference mode)
        if labels is None:
            return {"logits": logits}
        
        # Use attention mask as label mask if not provided
        if label_mask is None:
            label_mask = attention_mask
            
        # Calculate CRF loss
        loss = -self.crf(logits, labels, label_mask.bool(), reduction='mean')
        return {"loss": loss, "logits": logits}
    
    def decode(self, logits, mask):
        """Decode the best tag sequence using CRF."""
        return self.crf.decode(logits, mask.bool())


class MetricsPlotter:
    """Class for visualizing training metrics."""
    
    def __init__(self, output_dir: str):
        """Initialize with output directory for plots."""
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
    def plot_all(self, history: List[Dict[str, Any]]) -> None:
        """Generate all plots from training history."""
        self.plot_loss_curves(history)
        self.plot_token_f1_scores(history)
        self.plot_entity_f1_scores(history)
    
    def plot_loss_curves(self, history: List[Dict[str, Any]]) -> None:
        """Plot training and validation loss curves."""
        epochs = [entry["epoch"] for entry in history]
        train_losses = [entry["train_loss"] for entry in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-o', label="Training Loss")
        
        # Add validation loss if available
        if "val_metrics" in history[0] and "val_loss" in history[0]["val_metrics"]:
            val_losses = [entry["val_metrics"]["val_loss"] for entry in history]
            plt.plot(epochs, val_losses, 'r-^', label="Validation Loss")
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "loss.png"), dpi=300)
        plt.close()
    
    def plot_token_f1_scores(self, history: List[Dict[str, Any]]) -> None:
        """Plot token-level F1 scores."""
        if "val_metrics" not in history[0] or "token_level_micro_f1" not in history[0]["val_metrics"]:
            logger.warning("Token-level metrics not found in history. Skipping token F1 plot.")
            return
            
        epochs = [entry["epoch"] for entry in history]
        metrics = history[0]["val_metrics"]["token_level_micro_f1"]
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ["Binary", "3-Class", "7-Category"]
        
        for i, key in enumerate(["binary", "3-class", "7-category"]):
            if key in metrics:
                values = [entry["val_metrics"]["token_level_micro_f1"][key] for entry in history]
                plt.plot(epochs, values, marker='o', linestyle='-', 
                     color=colors[i], label=labels[i])
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.title("Token-Level Micro F1 Scores", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "token_metrics.png"), dpi=300)
        plt.close()
    
    def plot_entity_f1_scores(self, history: List[Dict[str, Any]]) -> None:
        """Plot entity-level F1 scores."""
        if "val_metrics" not in history[0] or "entity_level_micro_f1" not in history[0]["val_metrics"]:
            logger.warning("Entity-level metrics not found in history. Skipping entity F1 plot.")
            return
            
        epochs = [entry["epoch"] for entry in history]
        metrics = history[0]["val_metrics"]["entity_level_micro_f1"]
        
        plt.figure(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ["Binary", "3-Class", "7-Category"]
        
        for i, key in enumerate(["binary", "3-class", "7-category"]):
            if key in metrics:
                values = [entry["val_metrics"]["entity_level_micro_f1"][key] for entry in history]
                plt.plot(epochs, values, marker='o', linestyle='-', 
                     color=colors[i], label=labels[i])
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("F1 Score", fontsize=12)
        plt.title("Entity-Level Micro F1 Scores", fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, "entity_metrics.png"), dpi=300)
        plt.close()


class MedicalJargonIdentifier:
    """Model for identifying fine-grained medical jargon with CRF enhancement."""
    
    # Tag constants
    B_PREFIX = "B-"
    I_PREFIX = "I-"
    O_TAG = "O"
    
    # Label map based on the MEDREADME paper's 7-category taxonomy
    LABEL_MAP = {
        "O": 0,                                  # Non-jargon
        "B-medical-jargon-google-easy": 1,       # Beginning of Google-Easy jargon
        "I-medical-jargon-google-easy": 2,       # Inside of Google-Easy jargon
        "B-medical-jargon-google-hard": 3,       # Beginning of Google-Hard jargon
        "I-medical-jargon-google-hard": 4,       # Inside of Google-Hard jargon
        "B-medical-name-entity": 5,              # Beginning of medical name entity
        "I-medical-name-entity": 6,              # Inside of medical name entity
        "B-general-complex": 7,                  # Beginning of general complex term
        "I-general-complex": 8,                  # Inside of general complex term
        "B-abbr-medical": 9,                     # Beginning of medical abbreviation
        "I-abbr-medical": 10,                    # Inside of medical abbreviation
        "B-abbr-general": 11,                    # Beginning of general abbreviation
        "I-abbr-general": 12,                    # Inside of general abbreviation
        "B-general-medical-multisense": 13,      # Beginning of multi-sense word
        "I-general-medical-multisense": 14       # Inside of multi-sense word
    }
    
    def __init__(self, pretrained_model: str = "roberta-large", device: Optional[str] = None, 
                 max_length: int = 512):
        """Initialize the jargon identification model with CRF."""
        # Use GPU if available
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length
        
        # Initialize tokenizer with proper settings for word boundaries
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model, add_prefix_space=True)
        
        # Initialize transformer model
        transformer_model = AutoModelForTokenClassification.from_pretrained(pretrained_model)
        
        # Create our custom model with CRF layer
        self.model = CRFTokenClassificationModel(
            transformer_model=transformer_model,
            num_labels=len(self.LABEL_MAP),
            dropout_rate=0.1
        )
        self.model.to(self.device)
        
        # For tracking training
        self.training_history = []
        
        # Reverse mapping for decoding predictions
        self.ID_TO_LABEL = {v: k for k, v in self.LABEL_MAP.items()}
    
    def load_data(self, data_path: str, split: Optional[str] = None) -> List[Dict]:
        """Load jargon annotation data from JSON file, optionally filtering by split."""
        data = load_json(data_path)
        processed_examples = []
        
        for example in data:
            # Skip examples that don't match the requested split
            if split and example.get("split") != split:
                continue
                
            tokens = example.get("tokens", [])
            
            # Skip examples with no tokens
            if not tokens:
                logger.warning(f"Skipping example with no tokens: {example}")
                continue
                
            # Initialize all tokens as non-jargon ("O")
            labels = [self.O_TAG] * len(tokens)
            
            # Process the entities (jargon spans)
            self._process_entities(example.get("entities", []), tokens, labels)
            
            # Convert string labels to numeric IDs for the model
            label_ids = self._convert_labels_to_ids(labels)
            
            processed_examples.append({
                "tokens": tokens,
                "labels": label_ids,
                "text": " ".join(tokens),
                "split": example.get("split")
            })
        
        logger.info(f"Loaded {len(processed_examples)} examples" + 
                    (f" from split '{split}'" if split else ""))
        return processed_examples
    
    def _process_entities(self, entities: List, tokens: List[str], labels: List[str]) -> None:
        """Process entity spans and apply BIO tagging scheme."""
        for entity in entities:
            # Handle different entity formats
            if len(entity) < 3:
                logger.warning(f"Skipping entity with unexpected format: {entity}")
                continue
                
            start_idx, end_idx = entity[0], entity[1]
            entity_type = entity[2]
            
            # Ensure indices are within bounds
            if start_idx < 0 or end_idx > len(tokens) or start_idx >= end_idx:
                logger.warning(f"Skipping entity with invalid indices: {entity}")
                continue
                
            # First token gets "B-" prefix
            labels[start_idx] = f"{self.B_PREFIX}{entity_type.lower()}"
            
            # Subsequent tokens get "I-" prefix
            for i in range(start_idx + 1, end_idx):
                labels[i] = f"{self.I_PREFIX}{entity_type.lower()}"
    
    def _convert_labels_to_ids(self, labels: List[str]) -> List[int]:
        """Convert string labels to numeric IDs."""
        label_ids = []
        for label in labels:
            if label in self.LABEL_MAP:
                label_ids.append(self.LABEL_MAP[label])
            else:
                # Handle unknown labels by defaulting to "O"
                logger.warning(f"Unknown label: {label}, defaulting to '{self.O_TAG}'")
                label_ids.append(self.LABEL_MAP[self.O_TAG])
        return label_ids
    
    def prepare_dataset(self, examples: List[Dict], is_training: bool = True) -> JargonDataset:
        """Create dataset from processed examples."""
        return JargonDataset(
            examples, 
            self.tokenizer, 
            self.LABEL_MAP, 
            is_training,
            max_length=self.max_length
        )

    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        output_dir: str = "jargon_model",
        early_stopping: bool = True,
        patience: int = 3,
        gradient_accumulation_steps: int = 1
    ) -> List[Dict]:
        """Train the jargon identification model with CRF."""
        # Setup directories and optimizer
        os.makedirs(output_dir, exist_ok=True)
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # Calculate training parameters
        total_steps = len(train_dataloader) // gradient_accumulation_steps * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Setup optimizer with weight decay configuration
        optimizer_grouped_parameters = self._configure_optimizer_params(weight_decay)
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        
        # Setup learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training state variables
        global_step = 0
        best_f1 = 0.0  # Track best F1 score
        no_improvement_count = 0  # Counter for early stopping
        self.training_history = []
        
        logger.info(f"Starting training with {total_steps} total steps, {warmup_steps} warmup steps")
        
        # Main training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for step, batch in enumerate(progress_bar):
                # Process batch and compute loss
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    label_mask=batch["attention_mask"]
                )

                # Store original loss for logging
                original_loss = outputs["loss"].item()
                # Scale loss for gradient accumulation
                loss = outputs["loss"] / gradient_accumulation_steps
                loss.backward()

                epoch_loss += original_loss  # Use original loss for epoch average
                
                # Skip optimizer step if still accumulating gradients
                if (step + 1) % gradient_accumulation_steps != 0:
                    continue
                
                # Update weights
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # Log progress
                progress_bar.set_postfix({"loss": original_loss})
                
                # Regular evaluation
                if eval_dataloader and global_step % (len(train_dataloader) // 5) == 0:
                    metrics = self.evaluate(eval_dataloader)
                    
                    # Log and check for improvement
                    current_f1 = metrics["token_micro_f1"]
                    progress_bar.set_postfix({
                        "loss": original_loss,
                        "f1": current_f1
                    })
                    logger.info(f"Step {global_step}: token_micro_f1={current_f1:.4f}")
                    
                    # Check if model improved and save if it's better
                    model_improved = self._handle_model_checkpoint(
                        current_f1, best_f1, output_dir
                    )
                    
                    if model_improved:
                        best_f1 = current_f1
                    
                    # Back to training mode
                    self.model.train()
            
            # End of epoch processing
            avg_loss = epoch_loss / len(train_dataloader)
            logger.info(f"Average training loss for epoch {epoch+1}: {avg_loss:.4f}")
            
            # Evaluate at the end of each epoch
            if eval_dataloader:
                metrics = self.evaluate(eval_dataloader)
                current_f1 = metrics["token_micro_f1"]
                logger.info(f"End of epoch {epoch+1}: token_micro_f1={current_f1:.4f}")
                
                # Check if model improved
                model_improved = current_f1 > best_f1
                if model_improved:
                    best_f1 = current_f1
                    best_model_path = os.path.join(output_dir, "best_model")
                    self.save_model(best_model_path)
                    logger.info(f"New best model saved with F1: {best_f1:.4f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            
            # Store training history
            self.training_history.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                **({"val_metrics": metrics} if eval_dataloader else {})
            })
            
            # Early stopping check at epoch level - only if no step-level early stopping
            if early_stopping and no_improvement_count >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_model")
        self.save_model(final_model_path)
        logger.info(f"Saved final model to {final_model_path}")
        
        # Load best model if early stopping was used
        if early_stopping and best_f1 > 0:
            best_model_path = os.path.join(output_dir, "best_model")
            logger.info(f"Loading best model from {best_model_path}")
            self.load_model(best_model_path)
            
        # Save training history
        history_path = os.path.join(output_dir, "training_history.json")
        save_json(self.training_history, history_path)
        
        return self.training_history
    
    def _configure_optimizer_params(self, weight_decay: float) -> List[Dict]:
        """Configure parameters for optimizer with weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        return [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    
    def _handle_model_checkpoint(self, current_f1: float, best_f1: float, output_dir: str) -> bool:
        """Handle model checkpointing and return whether current model is best."""
        if current_f1 <= best_f1:
            return False
            
        best_model_path = os.path.join(output_dir, "best_model")
        self.save_model(best_model_path)
        logger.info(f"New best model saved with F1: {current_f1:.4f}")
        return True
    
    class EvaluationResults:
        """Data structure to hold evaluation results."""
        def __init__(self):
            self.token_predictions = []
            self.token_labels = []
            self.binary_token_predictions = []
            self.binary_token_labels = []
            self.three_class_token_predictions = []
            self.three_class_token_labels = []
            self.gold_entities = []
            self.pred_entities = []
            
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate the model on token and entity level metrics."""
        self.model.eval()
        
        # Initialize evaluation data
        eval_results = self.EvaluationResults()
        total_loss = 0.0
        num_batches = 0
        
        # Process each batch
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")
            for batch in progress_bar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Calculate loss for validation
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    label_mask=batch["attention_mask"]
                )
                total_loss += outputs["loss"].item()
                num_batches += 1
                
                # Get CRF predictions
                logits = outputs["logits"]
                predictions = self.model.decode(logits, batch["attention_mask"])
                
                # Process predictions for each example in batch
                self._process_evaluation_batch(
                    predictions, 
                    batch,
                    eval_results
                )
        
        # Calculate average validation loss
        val_loss = total_loss / max(num_batches, 1)
        
        # Check if predictions were generated
        if not eval_results.token_predictions:
            logger.warning("No token predictions generated during evaluation")
            empty_metrics = self._empty_evaluation_metrics()
            empty_metrics["val_loss"] = val_loss
            return empty_metrics
        
        # Compute metrics
        metrics = self._compute_evaluation_metrics(eval_results)
        metrics["val_loss"] = val_loss
        
        return metrics
    
    def _process_evaluation_batch(
        self, 
        predictions: List[List[int]], 
        batch: Dict[str, torch.Tensor],
        results: EvaluationResults
    ) -> None:
        """Process batch predictions for evaluation metrics."""
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        
        # Skip if no labels (inference mode)
        if labels is None:
            return
            
        # Process each example in the batch
        for i in range(len(predictions)):
            pred = predictions[i]
            gold = labels[i].cpu().numpy()
            valid_mask = attention_mask[i].cpu().bool().numpy()
            
            # Get valid tokens only (excluding padding)
            valid_length = sum(valid_mask)
            pred = pred[:valid_length]
            gold = gold[:valid_length]
            
            # Store token-level predictions and labels
            results.token_predictions.extend(pred)
            results.token_labels.extend(gold)
            
            # Create binary and 3-class groupings
            binary_pred, binary_gold = self._get_binary_labels(pred, gold)
            results.binary_token_predictions.extend(binary_pred)
            results.binary_token_labels.extend(binary_gold)
            
            three_class_pred, three_class_gold = self._get_three_class_labels(pred, gold)
            results.three_class_token_predictions.extend(three_class_pred)
            results.three_class_token_labels.extend(three_class_gold)
            
            # Convert token predictions to entity spans
            pred_entities = self._convert_bio_to_entities(pred)
            gold_entities = self._convert_bio_to_entities(gold)
            
            results.pred_entities.extend(pred_entities)
            results.gold_entities.extend(gold_entities)
    
    def _empty_evaluation_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure when no predictions are available."""
        return {
            "token_micro_f1": 0.0,
            "token_level": {
                "macro": {
                    "binary": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "3-class": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "7-category": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                },
                "micro": {
                    "binary": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "3-class": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "7-category": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                }
            },
            "entity_level": {
                "micro": {
                    "binary": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "3-class": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
                    "7-category": {"precision": 0.0, "recall": 0.0, "f1": 0.0}
                }
            },
            "token_level_micro_f1": {"binary": 0.0, "3-class": 0.0, "7-category": 0.0},
            "entity_level_micro_f1": {"binary": 0.0, "3-class": 0.0, "7-category": 0.0}
        }
    
    def _compute_evaluation_metrics(self, results: EvaluationResults) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics with consistent naming."""
        metrics = {}
        
        # Calculate token-level metrics for all granularities
        # 7-category (detailed)
        token_7cat_macro_p, token_7cat_macro_r, token_7cat_macro_f1, _ = precision_recall_fscore_support(
            results.token_labels, results.token_predictions, average='macro', zero_division=0
        )
        token_7cat_micro_p, token_7cat_micro_r, token_7cat_micro_f1, _ = precision_recall_fscore_support(
            results.token_labels, results.token_predictions, average='micro', zero_division=0
        )
        
        # Binary classification (jargon/non-jargon)
        token_binary_macro_p, token_binary_macro_r, token_binary_macro_f1, _ = precision_recall_fscore_support(
            results.binary_token_labels, results.binary_token_predictions, average='macro', zero_division=0
        )
        token_binary_micro_p, token_binary_micro_r, token_binary_micro_f1, _ = precision_recall_fscore_support(
            results.binary_token_labels, results.binary_token_predictions, average='micro', zero_division=0
        )
        
        # 3-class classification (medical/general/abbreviation)
        token_3class_macro_p, token_3class_macro_r, token_3class_macro_f1, _ = precision_recall_fscore_support(
            results.three_class_token_labels, results.three_class_token_predictions, 
            average='macro', zero_division=0
        )
        token_3class_micro_p, token_3class_micro_r, token_3class_micro_f1, _ = precision_recall_fscore_support(
            results.three_class_token_labels, results.three_class_token_predictions, 
            average='micro', zero_division=0
        )
        
        # Entity-level metrics (using custom functions)
        entity_7cat_results = self._compute_entity_level_metrics(
            results.gold_entities, results.pred_entities)
        entity_binary_results = self._compute_grouped_entity_metrics(
            results.gold_entities, results.pred_entities, "binary")
        entity_3class_results = self._compute_grouped_entity_metrics(
            results.gold_entities, results.pred_entities, "3-class")
        
        # Organize metrics in a consistent structure
        metrics["token_level"] = {
            "macro": {
                "binary": {
                    "precision": token_binary_macro_p,
                    "recall": token_binary_macro_r,
                    "f1": token_binary_macro_f1
                },
                "3-class": {
                    "precision": token_3class_macro_p,
                    "recall": token_3class_macro_r,
                    "f1": token_3class_macro_f1
                },
                "7-category": {
                    "precision": token_7cat_macro_p,
                    "recall": token_7cat_macro_r,
                    "f1": token_7cat_macro_f1
                }
            },
            "micro": {
                "binary": {
                    "precision": token_binary_micro_p,
                    "recall": token_binary_micro_r,
                    "f1": token_binary_micro_f1
                },
                "3-class": {
                    "precision": token_3class_micro_p,
                    "recall": token_3class_micro_r,
                    "f1": token_3class_micro_f1
                },
                "7-category": {
                    "precision": token_7cat_micro_p,
                    "recall": token_7cat_micro_r,
                    "f1": token_7cat_micro_f1
                }
            }
        }
        
        metrics["entity_level"] = {
            "micro": {
                "binary": entity_binary_results,
                "3-class": entity_3class_results,
                "7-category": entity_7cat_results
            }
        }
        
        # For backward compatibility, also provide the flattened shortcuts
        metrics["token_micro_f1"] = token_7cat_micro_f1
        metrics["token_binary_micro_f1"] = token_binary_micro_f1
        metrics["token_3class_micro_f1"] = token_3class_micro_f1
        
        # Summary metrics for easier access and comparison with paper results
        metrics["token_level_micro_f1"] = {
            "binary": token_binary_micro_f1,
            "3-class": token_3class_micro_f1,
            "7-category": token_7cat_micro_f1
        }
        
        metrics["entity_level_micro_f1"] = {
            "binary": entity_binary_results["f1"],
            "3-class": entity_3class_results["f1"],
            "7-category": entity_7cat_results["f1"]
        }
        
        return metrics
    
    def _get_binary_labels(self, pred: List[int], gold: List[int]) -> Tuple[List[int], List[int]]:
        """Convert detailed labels to binary (jargon/non-jargon)."""
        binary_pred = [1 if p > 0 else 0 for p in pred]  # 0 is "O" tag, anything else is jargon
        binary_gold = [1 if g > 0 else 0 for g in gold]
        return binary_pred, binary_gold
    
    def _get_three_class_labels(self, pred: List[int], gold: List[int]) -> Tuple[List[int], List[int]]:
        """Convert detailed labels to 3-class (medical/general/abbreviation)."""
        three_class_pred = []
        three_class_gold = []
        
        for p, g in zip(pred, gold):
            # Skip special tokens
            if p == -100 or g == -100:
                continue
                
            p_tag = self.ID_TO_LABEL.get(p, self.O_TAG)
            g_tag = self.ID_TO_LABEL.get(g, self.O_TAG)
            
            # Map to 3 classes: 0=O, 1=medical, 2=general, 3=abbreviation
            p_class = self._get_three_class_category(p_tag)
            g_class = self._get_three_class_category(g_tag)
            
            three_class_pred.append(p_class)
            three_class_gold.append(g_class)
            
        return three_class_pred, three_class_gold
    
    def _get_three_class_category(self, tag: str) -> int:
        """Map a tag to its 3-class category."""
        if tag == self.O_TAG:
            return 0
        if "medical-jargon" in tag or "medical-name" in tag:
            return 1
        if "general-complex" in tag or "general-medical-multisense" in tag:
            return 2
        if "abbr-" in tag:
            return 3
        return 0
    
    def _convert_bio_to_entities(self, bio_tags: List[int]) -> List[Dict[str, Any]]:
        """Convert BIO tag sequence to entity spans."""
        entities = []
        current_entity = None
        
        # First pass: extract entities
        for i, tag_id in enumerate(bio_tags):
            # Skip special tokens (-100)
            if tag_id == -100:
                continue
                
            # Get the tag string
            tag = self.ID_TO_LABEL.get(tag_id, self.O_TAG)
            
            # Handle B- tag (beginning of entity)
            if tag.startswith(self.B_PREFIX):
                # Close any open entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start a new entity
                current_entity = {
                    "type": tag[2:],  # Remove "B-" prefix
                    "start": i, 
                    "end": i + 1
                }
                continue
            
            # Handle I- tag (inside/continuation of entity)
            if tag.startswith(self.I_PREFIX):
                entity_type = tag[2:]  # Remove "I-" prefix
                
                # Extend current entity if types match
                if current_entity and current_entity["type"] == entity_type:
                    current_entity["end"] = i + 1
                    continue
                    
                # Start new entity if types don't match (invalid I- without B-)
                if current_entity is None or current_entity["type"] != entity_type:
                    current_entity = {
                        "type": entity_type,
                        "start": i, 
                        "end": i + 1
                    }
                continue
            
            # Handle O tag (outside any entity)
            if tag == self.O_TAG and current_entity:
                entities.append(current_entity)
                current_entity = None
        
        # Don't forget the last entity
        if current_entity:
            entities.append(current_entity)
        
        # Second pass: merge adjacent entities of the same type
        return self._merge_adjacent_entities(entities)
    
    def _merge_adjacent_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge adjacent entities of the same type."""
        if not entities:
            return []
            
        merged_entities = []
        current = entities[0]
        
        for i in range(1, len(entities)):
            next_entity = entities[i]
            # Check if entities are of the same type and adjacent
            if (current["type"] == next_entity["type"] and 
                next_entity["start"] - current["end"] <= 1):
                # Merge entities
                current["end"] = next_entity["end"]
            else:
                # Add current entity to merged list and move to next
                merged_entities.append(current)
                current = next_entity
        
        # Add the last entity
        merged_entities.append(current)
        
        return merged_entities
    
    def _compute_entity_level_metrics(self, gold_entities: List[Dict[str, Any]], 
                                    pred_entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute precision, recall, and F1 for entity-level evaluation."""
        tp = 0  # true positives
        pred_count = len(pred_entities)
        gold_count = len(gold_entities)
        
        if not pred_count or not gold_count:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Create a copy of gold_entities to track which ones have been matched
        matched_gold = [False] * len(gold_entities)
        
        # Count exact matches (same span boundaries and entity type)
        for pred in pred_entities:
            for i, gold in enumerate(gold_entities):
                if (not matched_gold[i] and
                    pred["start"] == gold["start"] and 
                    pred["end"] == gold["end"] and 
                    pred["type"] == gold["type"]):
                    tp += 1
                    matched_gold[i] = True  # Mark this gold entity as matched
                    break
        
        # Calculate precision, recall, and F1
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gold_count if gold_count else 0.0
        
        if precision + recall == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            
        f1 = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1": f1}
    
    def _compute_grouped_entity_metrics(self, gold_entities: List[Dict[str, Any]], 
                                      pred_entities: List[Dict[str, Any]], 
                                      mode: str) -> Dict[str, float]:
        """Compute micro F1 for grouped entity types (binary or 3-class)."""
        # Group entity types based on mode
        if mode == "binary":
            # Binary: jargon vs non-jargon
            gold_entities = [{"start": e["start"], "end": e["end"], "type": "jargon"} 
                           for e in gold_entities]
            pred_entities = [{"start": e["start"], "end": e["end"], "type": "jargon"} 
                            for e in pred_entities]
        elif mode == "3-class":
            # 3-class: medical / general / abbreviation
            gold_entities = [{"start": e["start"], "end": e["end"], 
                            "type": self._get_entity_group(e["type"])} 
                           for e in gold_entities]
            pred_entities = [{"start": e["start"], "end": e["end"], 
                            "type": self._get_entity_group(e["type"])} 
                            for e in pred_entities]
        
        # Filter out duplicates (same span and type)
        gold_entities = self._filter_duplicates(gold_entities)
        pred_entities = self._filter_duplicates(pred_entities)
        
        # Compute metrics for grouped entities
        return self._compute_entity_level_metrics(gold_entities, pred_entities)
    
    def _get_entity_group(self, entity_type: str) -> str:
        """Map detailed entity type to high-level group."""
        if "medical-jargon" in entity_type or "medical-name" in entity_type:
            return "medical"
        if entity_type.startswith("abbr"):
            return "abbreviation"
        return "general"
    
    def _filter_duplicates(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities (same span and type)."""
        unique_entities = []
        seen: Set[Tuple[int, int, str]] = set()
        
        for entity in entities:
            key = (entity["start"], entity["end"], entity["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict jargon spans in text using CRF decoding."""
        # Handle empty input
        if not text or not text.strip():
            logger.warning("Empty text provided for prediction")
            return {"text": text, "jargon_spans": []}
        
        # Use batch prediction with a single text
        results = self.predict_batch([text])
        return results[0]

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict jargon spans for a batch of texts using true batching."""
        if not texts:
            return []
            
        # Set model to evaluation mode
        self.model.eval()
        logger.info(f"Predicting jargon spans for {len(texts)} texts...")
        
        # Calculate the actual number of batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        results = []
        
        # Process in batches with progress bar
        for batch_idx in tqdm(range(num_batches), desc="Predicting jargon spans"):
            # Get the current batch of texts
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            
            # Get token predictions for this batch
            batch_token_predictions, batch_offset_mappings, batch_word_ids = self._get_token_predictions_crf(batch_texts)
            
            # Process predictions for each text in the batch
            batch_results = []
            for i, text in enumerate(batch_texts):
                # Convert token predictions to word-level tags
                word_tags = self._aggregate_to_word_level(
                    batch_token_predictions[i], 
                    batch_offset_mappings[i], 
                    batch_word_ids[i]
                )
                
                # Extract jargon spans from tags
                jargon_spans = self._get_entity_spans_from_tags(word_tags, text)
                
                # Add results for this text
                batch_results.append({
                    "text": text,
                    "jargon_spans": jargon_spans
                })
            
            # Add batch results to the overall results
            results.extend(batch_results)
            
            # Free up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results

    def _get_token_predictions_crf(self, texts: List[str]) -> Tuple[List[List[str]], List[np.ndarray], List[List[Optional[int]]]]:
        """Get token-level predictions using CRF decoding for a batch of texts."""
        # Batch tokenize all texts together
        encodings = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True
        )
        
        # Extract offset mappings and word IDs for each text
        batch_offset_mappings = [encodings.offset_mapping[i].numpy() for i in range(len(texts))]
        batch_word_ids = [encodings.word_ids(i) for i in range(len(texts))]
        
        # Move tensors to device
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)
        
        # Run inference with CRF decoding
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs["logits"]
            batch_predictions = self.model.decode(logits, attention_mask)
        
        # Convert numeric predictions to tags for each text
        batch_token_tags = []
        for predictions in batch_predictions:
            token_tags = [self.ID_TO_LABEL.get(pred, self.O_TAG) for pred in predictions]
            batch_token_tags.append(token_tags)
        
        return batch_token_tags, batch_offset_mappings, batch_word_ids

    def _aggregate_to_word_level(
        self, 
        token_tags: List[str], 
        offset_mapping: np.ndarray, 
        word_ids: List[Optional[int]]
    ) -> List[Tuple[int, str, Tuple[int, int]]]:
        """Aggregate subword token predictions to word level."""
        logger.debug(f"Text tokenized into {len(token_tags)} tokens")
        
        # Maps to store word-level information
        word_to_chars: Dict[int, Tuple[int, int]] = {}  # word_id → (start_char, end_char)
        word_to_tags: Dict[int, List[str]] = {}         # word_id → [tags]
        
        # Group token information by word_id
        for i, (tag, (start_char, end_char), word_idx) in enumerate(zip(token_tags, offset_mapping, word_ids)):
            if word_idx is None:  # Skip special tokens
                continue
                
            if word_idx not in word_to_chars:
                word_to_chars[word_idx] = (start_char, end_char)
                word_to_tags[word_idx] = [tag]
            else:
                _, prev_end = word_to_chars[word_idx]
                word_to_chars[word_idx] = (word_to_chars[word_idx][0], max(prev_end, end_char))
                word_to_tags[word_idx].append(tag)
        
        # Process words in order
        word_tags = []
        for word_idx in sorted(word_to_tags.keys()):
            tags = word_to_tags[word_idx]
            final_tag = self._vote_for_tag(tags)
            word_tags.append((word_idx, final_tag, word_to_chars[word_idx]))
        
        return word_tags
    
    def _vote_for_tag(self, tags: List[str]) -> str:
        """Vote for the best tag with priority: B- > I- > O."""
        # Count occurrences of each tag type
        tag_counts: Dict[str, int] = {}
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Determine the tag with the highest count, prioritizing B > I > O
        b_tags = {t: c for t, c in tag_counts.items() if t.startswith(self.B_PREFIX)}
        i_tags = {t: c for t, c in tag_counts.items() if t.startswith(self.I_PREFIX)}
        
        if b_tags:
            # Use the most common B-tag
            return max(b_tags.items(), key=lambda x: x[1])[0]
            
        if i_tags:
            # Use the most common I-tag
            return max(i_tags.items(), key=lambda x: x[1])[0]
            
        # Otherwise, it's an O tag
        return self.O_TAG
    
    def _get_entity_spans_from_tags(self, word_tags: List[Tuple[int, str, Tuple[int, int]]], text: str) -> List[Dict[str, Any]]:
        """Convert word-level BIO tags to entity spans."""
        jargon_spans = []
        current_entity = None
        
        # Extract entities from tags
        for word_idx, tag, (start_char, end_char) in word_tags:
            # Skip very short spans (e.g., single punctuation marks)
            char_span = text[start_char:end_char].strip()
            if end_char - start_char <= 1 and char_span in ".,:;-()[]{}\"'":
                continue
                
            # Process different tag types
            if tag.startswith(self.B_PREFIX):
                # Close previous entity
                if current_entity:
                    entity_text = text[current_entity["start"]:current_entity["end"]]
                    if self._is_valid_entity(entity_text):
                        current_entity["text"] = entity_text
                        jargon_spans.append(current_entity)
                
                # Start new entity
                current_entity = {
                    "type": tag[2:],
                    "start": start_char,
                    "end": end_char
                }
            elif tag.startswith(self.I_PREFIX):
                entity_type = tag[2:]
                
                # Continue current entity if types match
                if current_entity and current_entity["type"] == entity_type:
                    current_entity["end"] = end_char
                # Start new entity if types don't match (invalid I- without B-)
                elif current_entity is None or current_entity["type"] != entity_type:
                    current_entity = {
                        "type": entity_type,
                        "start": start_char,
                        "end": end_char
                    }
            elif tag == self.O_TAG and current_entity:
                # Close current entity
                entity_text = text[current_entity["start"]:current_entity["end"]]
                if self._is_valid_entity(entity_text):
                    current_entity["text"] = entity_text
                    jargon_spans.append(current_entity)
                current_entity = None
        
        # Handle the last entity if there is one
        if current_entity:
            entity_text = text[current_entity["start"]:current_entity["end"]]
            if self._is_valid_entity(entity_text):
                current_entity["text"] = entity_text
                jargon_spans.append(current_entity)
        
        # Post-process to merge adjacent entities
        return self._post_process_spans(jargon_spans, text)
    
    def _post_process_spans(self, spans: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Post-process spans to merge adjacent entities and remove duplicates."""
        if not spans:
            return []
            
        # Sort by start position for processing
        spans.sort(key=lambda x: x["start"])
        
        merged_spans = []
        current_span = spans[0].copy()
        
        for i in range(1, len(spans)):
            next_span = spans[i]
            
            # Check if spans should be merged (same type and close together)
            if (current_span["type"] == next_span["type"] and 
                next_span["start"] - current_span["end"] <= 2):
                
                # Check if there's only whitespace or a hyphen between entities
                between_text = text[current_span["end"]:next_span["start"]]
                if between_text.strip() in ["", "-"]:
                    # Merge the spans
                    current_span["end"] = next_span["end"]
                    current_span["text"] = text[current_span["start"]:current_span["end"]]
                    continue
            
            # Different type or not adjacent, add current span and start next one
            merged_spans.append(current_span)
            current_span = next_span.copy()
        
        # Add the last span
        merged_spans.append(current_span)
        
        return merged_spans

    def _is_valid_entity(self, entity_text: str) -> bool:
        """Check if an entity text is valid."""
        # Skip empty entities
        if not entity_text.strip():
            return False
            
        # Skip single punctuation
        if len(entity_text.strip()) <= 1 and entity_text.strip() in ".,:;-()[]{}\"'":
            return False
            
        # Skip very short entities
        if len(entity_text.strip()) <= 1:
            return False
            
        return True
    
    def save_model(self, output_dir: str) -> None:
        """Save the model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        # Save model weights
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # Save model config and label mapping
        config_dict = {
            "label_map": self.LABEL_MAP,
            "pretrained_model_name": self.tokenizer.name_or_path,
            "max_length": self.max_length
        }
        save_json(config_dict, os.path.join(output_dir, "config.json"))
        
        logger.info(f"Saved model and tokenizer to {output_dir}")
    
    def load_model(self, model_dir: str) -> bool:
        """Load a saved model."""
        if not os.path.exists(model_dir):
            logger.error(f"Model directory {model_dir} does not exist")
            return False
        
        # Load config
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            config = load_json(config_path)
            self.LABEL_MAP = config.get("label_map", self.LABEL_MAP)
            self.ID_TO_LABEL = {v: k for k, v in self.LABEL_MAP.items()}
            self.max_length = config.get("max_length", self.max_length)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load base model
        pretrained_name = config.get("pretrained_model_name", "roberta-large")
        transformer_model = AutoModelForTokenClassification.from_pretrained(
            pretrained_name, 
            num_labels=len(self.LABEL_MAP)
        )
        
        # Initialize CRF model
        self.model = CRFTokenClassificationModel(
            transformer_model=transformer_model,
            num_labels=len(self.LABEL_MAP),
            dropout_rate=0.1
        )
        
        # Load model weights
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(model_path):
            logger.error(f"Model weights file not found at {model_path}")
            return False
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        logger.info(f"Successfully loaded model weights from {model_path}")
        return True


def create_data_loaders(train_dataset: Dataset, eval_dataset: Optional[Dataset] = None, batch_size: int = 8) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create data loaders for training and evaluation."""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    if eval_dataset is None:
        return train_dataloader, None
        
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, eval_dataloader


def train_jargon_model_crf(
    data_path: str, 
    output_dir: str = "jargon_model_crf", 
    pretrained_model: str = "roberta-large",
    batch_size: int = 16,
    epochs: int = 15,
    learning_rate: float = 2e-6,
    early_stopping: bool = True,
    patience: int = 3,
    gradient_accumulation_steps: int = 2,
    max_length: int = 512,
    warmup_ratio: float = 0.1
) -> Tuple[Optional[MedicalJargonIdentifier], Optional[Dict[str, Any]]]:
    """Train and evaluate the jargon identification model with CRF."""
    # Create the model with CRF
    jargon_model = MedicalJargonIdentifier(
        pretrained_model=pretrained_model,
        max_length=max_length
    )
    
    # Load data using the predefined splits
    train_examples = jargon_model.load_data(data_path, split="train")
    dev_examples = jargon_model.load_data(data_path, split="dev")
    test_examples = jargon_model.load_data(data_path, split="test")
    
    # Validate data
    if not train_examples:
        logger.error("No training examples found. Please check the data path and format.")
        return None, None
        
    # Log dataset statistics
    logger.info(f"Dataset statistics:")
    logger.info(f"  Training examples: {len(train_examples)}")
    logger.info(f"  Development examples: {len(dev_examples)}")
    logger.info(f"  Test examples: {len(test_examples)}")
    
    # Create datasets and dataloaders
    train_dataset = jargon_model.prepare_dataset(train_examples, is_training=True)
    dev_dataset = jargon_model.prepare_dataset(dev_examples, is_training=True)
    test_dataset = jargon_model.prepare_dataset(test_examples, is_training=True)
    
    train_dataloader, dev_dataloader = create_data_loaders(
        train_dataset, dev_dataset, batch_size=batch_size
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Train the model
    jargon_model.train(
        train_dataloader=train_dataloader,
        eval_dataloader=dev_dataloader,
        epochs=epochs,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        output_dir=output_dir,
        early_stopping=early_stopping,
        patience=patience,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    # Create plots of training metrics
    metrics_plotter = MetricsPlotter(output_dir)
    metrics_plotter.plot_all(jargon_model.training_history)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set:")
    test_metrics = jargon_model.evaluate(test_dataloader)
    
    # Print formatted metrics
    print("\n---- Token-Level Micro F1 Scores with CRF ----")
    print("Binary: {:.1f}".format(test_metrics["token_level_micro_f1"]["binary"] * 100))
    print("3-Class: {:.1f}".format(test_metrics["token_level_micro_f1"]["3-class"] * 100))
    print("7-Category: {:.1f}".format(test_metrics["token_level_micro_f1"]["7-category"] * 100))
    
    print("\n---- Entity-Level Micro F1 Scores with CRF ----")
    print("Binary: {:.1f}".format(test_metrics["entity_level_micro_f1"]["binary"] * 100))
    print("3-Class: {:.1f}".format(test_metrics["entity_level_micro_f1"]["3-class"] * 100))
    print("7-Category: {:.1f}".format(test_metrics["entity_level_micro_f1"]["7-category"] * 100))
    
    # Save test metrics
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    save_json(test_metrics, metrics_path)
    
    return jargon_model, test_metrics


def main():
    """Demonstrate using the improved jargon identification model and classifier with batch processing."""
    import time
    
    # Configure model paths
    model_path = "/data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme"
    
    # Load pre-trained model for inference
    print("Loading jargon identification model...")
    jargon_identifier = MedicalJargonIdentifier(pretrained_model="roberta-large")
    best_model_path = os.path.join(model_path, "best_model")
    jargon_identifier.load_model(best_model_path)
    
    # Initialize classifier
    print("Initializing jargon classifier...")
    jargon_classifier = MedicalJargonClassifier(
        jargon_model=jargon_identifier,
        spacy_model="en_core_web_sm"  # Using smaller model for faster loading
    )
    
    # Example texts with varying complexity
    example_texts = [
        # Easy medical text
        "The patient is taking ibuprofen for pain relief. Blood pressure is normal.",
        
        # Moderate medical jargon
        "An oro-antral communication is an unnatural opening between the oral cavity and maxillary sinus.",
        
        # Complex medical text with a mix of jargon types
        "Furuncles and carbuncles, caused primarily by Staphylococcus aureus infection, demonstrate high curability with appropriate intervention.",
        
        # Very complex medical text
        "Together, these findings reveal the physiological role for KMT5c-mediated H4K20 methylation in the maintenance and activation of the thermo-genic program in adipocytes.",
        
        # Text with abbreviations
        "The patient's ECG showed signs of AF, and the CBC indicated mild anemia. CRP levels were elevated.",
    ]
    
    # Test with different batch sizes to demonstrate batch processing
    batch_sizes = [None, 2]
    
    for batch_size in batch_sizes:
        print(f"\n===== PROCESSING WITH {'ALL TEXTS AT ONCE' if batch_size is None else f'BATCH SIZE {batch_size}'} =====")
        
        # Measure processing time
        start_time = time.time()
        results = jargon_classifier.predict_batch(example_texts, batch_size=batch_size)
        processing_time = time.time() - start_time
        
        print(f"Processing time: {processing_time:.3f} seconds\n")
        
        # Display key metrics for each text
        for i, text in enumerate(example_texts):
            metrics = results[i]
            
            print(f"Text {i+1}: {text[:50]}..." if len(text) > 50 else f"Text {i+1}: {text}")
            
            # Print the most important metrics
            print(f"  Key Metrics:")
            print(f"  - Jargon count: {metrics['jargon_count']}")
            print(f"  - Jargon density: {metrics['jargon_density']:.2f} per 100 words")
            print(f"  - Medical density: {metrics['medical_density']:.2f} per 100 words")
            print(f"  - General density: {metrics['general_density']:.2f} per 100 words")
            print(f"  - Abbreviation density: {metrics['abbreviation_density']:.2f} per 100 words")
            print(f"  - Hard-to-easy ratio: {metrics['hard_to_easy_ratio']:.2f}")
            print(f"  - Avg sentence jargon density: {metrics['avg_sentence_jargon_density']:.2f}")
            print(f"  - Jargon diversity: {metrics['jargon_diversity']:.2f}")
            print(f"  - Jargon clustering index: {metrics['jargon_clustering_index']:.4f}")
            
            # Display jargon distribution by category
            print("  Jargon distribution:")
            for category in jargon_classifier.ALL_CATEGORIES:
                density = metrics[f"{category.replace('-', '_')}_density"]
                if density > 0:
                    print(f"    - {category}: {density:.2f} per 100 words")
            
            print()
    
    # Compare with individual processing (to demonstrate the performance improvement)
    print("\n===== PROCESSING TEXTS INDIVIDUALLY =====")
    start_time = time.time()
    individual_results = [jargon_classifier.predict_single(text) for text in example_texts]
    individual_time = time.time() - start_time
    print(f"Processing time: {individual_time:.3f} seconds")
    
    # Calculate speedup
    if batch_sizes:
        best_batch_time = min(processing_time, individual_time)
        speedup = individual_time / best_batch_time
        print(f"\nBatch processing speedup: {speedup:.2f}x faster than individual processing")


if __name__ == "__main__":
    main()