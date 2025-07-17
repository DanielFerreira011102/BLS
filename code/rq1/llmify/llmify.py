import os
import json
import glob
import argparse
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

from utils.helpers import setup_logging, load_json, save_json

# Initialize logging
logger = setup_logging()

# Model configuration constants
DIMENSIONS = [
    "vocabulary_complexity",
    "syntactic_complexity", 
    "conceptual_density",
    "background_knowledge",
    "cognitive_load"
]

# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_data(file_patterns: List[str]) -> List[Dict]:
    """Load readability data from JSON files matching the given glob patterns."""
    all_file_paths = _collect_file_paths(file_patterns)
    dataset_files = _load_dataset_files(all_file_paths)
    return _process_datasets(dataset_files)

def _collect_file_paths(patterns: List[str]) -> List[str]:
    """Collect all file paths matching the given patterns."""
    all_paths = []
    for pattern in patterns:
        all_paths.extend(glob.glob(pattern, recursive=True))
    
    logger.info(f"Found {len(all_paths)} files matching the provided patterns")
    return all_paths

def _load_dataset_files(file_paths: List[str]) -> Dict[str, List[Tuple[str, List]]]:
    """Load and organize dataset files by dataset name."""
    dataset_files = {}
    
    for file_path in file_paths:
        data = load_json(file_path)
        
        for dataset_name, dataset_data in data.items():
            dataset = dataset_data.get('dataset') or dataset_name
            model = os.path.basename(dataset_data['config']['llm_model_path'])
            
            if dataset not in dataset_files:
                dataset_files[dataset] = []
            dataset_files[dataset].append((model, dataset_data['samples']))
    
    return dataset_files

def _process_datasets(dataset_files: Dict[str, List[Tuple[str, List]]]) -> List[Dict]:
    """Process datasets into standardized format."""
    all_data = []
    
    for dataset, model_samples in dataset_files.items():
        processed_samples = _process_single_dataset(dataset, model_samples)
        all_data.extend(processed_samples)
    
    logger.info(f"Loaded {len(all_data)} samples with scores from multiple models")
    return all_data

def _process_single_dataset(dataset: str, model_samples: List[Tuple[str, List]]) -> List[Dict]:
    """Process a single dataset's samples."""
    # Sort by model name for consistency
    model_samples.sort(key=lambda x: x[0])
    
    # Skip empty datasets
    if not model_samples or not model_samples[0][1]:
        return []
    
    samples = []
    num_samples = len(model_samples[0][1])
    
    for sample_idx in range(num_samples):
        # Skip if sample index is out of bounds for any model
        if any(sample_idx >= len(model_data) for _, model_data in model_samples):
            continue
        
        # Process both text types
        for text_type in ["simple", "expert"]:
            sample = _create_sample(dataset, model_samples, sample_idx, text_type)
            if sample:
                samples.append(sample)
    
    return samples

def _create_sample(dataset: str, model_samples: List[Tuple[str, List]], 
                  sample_idx: int, text_type: str) -> Optional[Dict]:
    """Create a single sample with text and scores."""
    text = model_samples[0][1][sample_idx][text_type]["text"]
    
    # Gather scores from all models
    all_scores = []
    for model, samples in model_samples:
        score_dict = samples[sample_idx][text_type]["metrics"]["llm"]["dimension_scores"]
        all_scores.append((model, score_dict))
    
    return {
        "text": text,
        "dataset": dataset,
        "type": text_type,
        "scores": all_scores
    }

# ============================================================================
# Dataset Class
# ============================================================================

class ReadabilityDataset(Dataset):
    """PyTorch Dataset for readability scoring with multiple loss function support."""
    
    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512, 
                 model_filters: Optional[Dict] = None, soft_dist_type: str = 'squared_distance', 
                 temperature: float = 1.0):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_filters = model_filters or {}
        self.soft_dist_type = soft_dist_type.lower()
        self.temperature = temperature
        
        self._validate_soft_dist_type()
        self._precompute_targets()
    
    def _validate_soft_dist_type(self):
        """Validate soft distribution type parameter."""
        valid_types = ['squared_distance', 'histogram']
        if self.soft_dist_type not in valid_types:
            raise ValueError(f"soft_dist_type must be one of {valid_types}")
    
    def _precompute_targets(self):
        """Precompute targets for each loss function to avoid repeated computation."""
        self.mse_targets = []
        self.ce_targets = []
        self.soft_ce_targets = []
        self.kl_targets = []
        
        for sample in self.samples:
            targets = self._compute_sample_targets(sample)
            self.mse_targets.append(targets['mse'])
            self.ce_targets.append(targets['ce'])
            self.soft_ce_targets.append(targets['soft_ce'])
            self.kl_targets.append(targets['kl'])
    
    def _compute_sample_targets(self, sample: Dict) -> Dict:
        """Compute all target types for a single sample."""
        scores = sample['scores']
        
        # Filter scores for each loss type
        filtered_scores = {
            'mse': self._filter_scores(scores, self.model_filters.get('mse')),
            'ce': self._filter_scores(scores, self.model_filters.get('ce')),
            'soft_ce': self._filter_scores(scores, self.model_filters.get('soft_ce')),
            'kl': self._filter_scores(scores, self.model_filters.get('kl'))
        }
        
        targets = {}
        for loss_type, loss_scores in filtered_scores.items():
            targets[loss_type] = self._compute_targets_for_loss_type(loss_scores, loss_type)
        
        return targets
    
    def _compute_targets_for_loss_type(self, scores: List[Tuple], loss_type: str) -> List:
        """Compute targets for a specific loss type."""
        targets = []
        
        for dim in DIMENSIONS:
            dim_scores = [score_dict[dim] for _, score_dict in scores]
            
            if loss_type == 'mse':
                target = sum(dim_scores) / max(len(dim_scores), 1)
            elif loss_type == 'ce':
                avg_score = sum(dim_scores) / max(len(dim_scores), 1)
                target = round(avg_score) - 1  # Convert to 0-indexed
            else:  # soft_ce or kl
                target = self._create_soft_distribution(dim_scores)
            
            targets.append(target)
        
        return targets
    
    def _filter_scores(self, scores: List[Tuple], model_filter: Optional[List[str]]) -> List[Tuple]:
        """Filter scores by model names if filter is provided."""
        if not model_filter:
            return scores
        
        filtered = [(model, score_dict) for model, score_dict in scores 
                   if model in model_filter]
        return filtered if filtered else scores  # Fallback to all scores
    
    def _create_soft_distribution(self, scores: List[float]) -> torch.Tensor:
        """Create soft distribution from scores with temperature scaling."""
        if not scores:
            return torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])  # Uniform distribution
        
        if self.soft_dist_type == 'squared_distance':
            return self._squared_distance_distribution(scores)
        
        return self._histogram_distribution(scores)
    
    def _squared_distance_distribution(self, scores: List[float]) -> torch.Tensor:
        """Create distribution using negative squared distances."""
        distances = torch.zeros(5)
        for score in scores:
            for i in range(5):
                distances[i] += -((i + 1 - score) ** 2)
        return F.softmax(distances / self.temperature, dim=0)
    
    def _histogram_distribution(self, scores: List[float]) -> torch.Tensor:
        """Create simple histogram-like distribution."""
        dist = torch.zeros(5)
        for score in scores:
            idx = min(max(round(score) - 1, 0), 4)  # Map 1-5 to 0-4
            dist[idx] += 1
        return F.softmax(dist / self.temperature, dim=0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with tokenized text and precomputed targets."""
        sample = self.samples[idx]
        
        # Tokenize text
        encoded = self.tokenizer(
            sample['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'mse_target': torch.tensor(self.mse_targets[idx], dtype=torch.float),
            'ce_target': torch.tensor(self.ce_targets[idx], dtype=torch.long),
            'soft_ce_target': torch.stack(self.soft_ce_targets[idx]),
            'kl_target': torch.stack(self.kl_targets[idx])
        }

# ============================================================================
# Model Architecture
# ============================================================================

class BaseReadabilityModel(nn.Module):
    """Base class for readability models with common functionality."""
    
    def __init__(self, model_name: str, tokenizer=None, num_dimensions: int = len(DIMENSIONS), 
                 num_scores: int = 5):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.num_dimensions = num_dimensions
        self.num_scores = num_scores
    
    def _ensure_tokenizer(self):
        """Load tokenizer if not already set."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def predict_batch(self, texts: List[str], batch_size: int = 8, 
                     max_length: int = 512) -> List[Dict]:
        """Predict readability scores for a batch of texts."""
        self._ensure_tokenizer()
        device = next(self.parameters()).device
        self.eval()
        
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self._process_batch(batch_texts, device, max_length)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, batch_texts: List[str], device: torch.device, 
                      max_length: int) -> List[Dict]:
        """Process a single batch of texts."""
        # Tokenize batch
        encoded = self.tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device and perform forward pass
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = self(input_ids, attention_mask)
            scores = self._compute_expected_scores(logits)
        
        # Format results
        return [self._format_result(scores[j]) for j in range(len(batch_texts))]
    
    def _compute_expected_scores(self, logits: torch.Tensor) -> np.ndarray:
        """Compute expected scores from logits using softmax probabilities."""
        probs = F.softmax(logits, dim=2)
        possible_values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, device=logits.device)
        return (probs * possible_values.view(1, 1, 5)).sum(dim=2).cpu().numpy()
    
    def _format_result(self, scores: np.ndarray) -> Dict:
        """Format prediction results for a single text."""
        dimension_scores = {DIMENSIONS[k]: float(scores[k]) for k in range(len(DIMENSIONS))}
        overall_score = float(np.mean(scores))
        
        return {
            "dimension_scores": dimension_scores,
            "overall_score": overall_score
        }
    
    def predict_single(self, text: str, max_length: int = 512) -> Dict:
        """Predict readability scores for a single text."""
        return self.predict_batch([text], batch_size=1, max_length=max_length)[0]
    
    def predict(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> List[Dict]:
        """Predict readability scores for a list of texts (compatibility wrapper)."""
        return self.predict_batch(texts, batch_size, max_length)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the forward method")

class ReadabilityModel(BaseReadabilityModel):
    """BERT-based model with shared representation and dimension-specific heads."""
    
    def __init__(self, model_name: str, tokenizer=None, num_dimensions: int = len(DIMENSIONS), 
                 num_scores: int = 5, dropout_rate: float = 0.2):
        super().__init__(model_name, tokenizer, num_dimensions, num_scores)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dimension-specific classification heads
        self.dimension_heads = nn.ModuleList([
            nn.Linear(hidden_size // 2, num_scores) for _ in range(num_dimensions)
        ])
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through BERT and dimension-specific heads."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply shared representation
        shared_repr = self.shared_layer(pooled_output)
        
        # Apply dimension-specific heads and stack results
        logits = torch.stack([head(shared_repr) for head in self.dimension_heads], dim=1)
        
        return logits

class ImprovedReadabilityModel(BaseReadabilityModel):
    """Advanced BERT-based model with attention mechanisms and cross-dimension interactions."""
    
    def __init__(self, model_name: str, tokenizer=None, num_dimensions: int = len(DIMENSIONS), 
                 num_scores: int = 5, attention_heads: int = 2, dropout_rate: float = 0.2):
        super().__init__(model_name, tokenizer, num_dimensions, num_scores)
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Document-level attention for weighted token representations
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Shared representation layers with layer normalization
        self.shared_layer1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.shared_layer2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Dimension-specific feature projections
        self.dimension_projections = nn.ModuleList([
            nn.Linear(hidden_size // 2, hidden_size // 4) for _ in range(num_dimensions)
        ])
        
        # Cross-dimension attention for capturing interactions
        self.dimension_attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 4,
            num_heads=attention_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification heads for each dimension
        self.classification_heads = nn.ModuleList([
            nn.Linear(hidden_size // 4, num_scores) for _ in range(num_dimensions)
        ])
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced feature extraction and attention mechanisms."""
        # Get BERT token representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_reprs = outputs.last_hidden_state
        
        # Compute document-level representation using attention
        doc_repr = self._compute_document_representation(token_reprs, attention_mask)
        
        # Process through shared layers with residual connections
        doc_repr = self._apply_shared_layers(doc_repr)
        
        # Create dimension-specific features and apply cross-attention
        dimension_features = self._create_dimension_features(doc_repr)
        
        # Apply classification heads
        return self._apply_classification_heads(dimension_features)
    
    def _compute_document_representation(self, token_reprs: torch.Tensor, 
                                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute weighted document representation using attention."""
        attention_scores = self.word_attention(token_reprs)
        attention_mask_expanded = attention_mask.unsqueeze(-1)
        
        # Mask invalid positions
        attention_scores = attention_scores.masked_fill(attention_mask_expanded == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute weighted sum
        return torch.sum(token_reprs * attention_weights, dim=1)
    
    def _apply_shared_layers(self, doc_repr: torch.Tensor) -> torch.Tensor:
        """Apply shared layers with residual connection."""
        residual = doc_repr
        doc_repr = self.shared_layer1(doc_repr)
        doc_repr = doc_repr + residual  # Residual connection
        return self.shared_layer2(doc_repr)
    
    def _create_dimension_features(self, doc_repr: torch.Tensor) -> torch.Tensor:
        """Create dimension-specific features with cross-attention."""
        # Project to dimension-specific features
        dimension_features = torch.stack([
            proj(doc_repr) for proj in self.dimension_projections
        ], dim=1)
        
        # Apply cross-dimension attention
        attn_output, _ = self.dimension_attention(
            dimension_features, dimension_features, dimension_features
        )
        
        # Add residual connection
        return dimension_features + attn_output
    
    def _apply_classification_heads(self, dimension_features: torch.Tensor) -> torch.Tensor:
        """Apply classification heads to dimension features."""
        return torch.stack([
            self.classification_heads[i](dimension_features[:, i]) 
            for i in range(self.num_dimensions)
        ], dim=1)

# ============================================================================
# Loss Functions
# ============================================================================

def mse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss between predicted and target scores."""
    probs = F.softmax(logits, dim=2)
    possible_values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float, device=probs.device)
    expected_scores = (probs * possible_values.view(1, 1, 5)).sum(dim=2)
    return F.mse_loss(expected_scores, targets.to(logits.device))

def kl_divergence_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """KL divergence between predicted and target distributions."""
    log_distributions = F.log_softmax(logits, dim=2)
    targets = targets.to(logits.device)
    return F.kl_div(log_distributions, targets, reduction='batchmean')

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss treating scores as class labels."""
    targets = targets.to(logits.device)
    
    loss = 0
    for i in range(logits.size(1)):
        loss += F.cross_entropy(logits[:, i], targets[:, i])
    
    return loss / logits.size(1)

def soft_cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft cross-entropy using target distributions."""
    log_distributions = F.log_softmax(logits, dim=2)
    targets = targets.to(logits.device)
    return -torch.mean(torch.sum(targets * log_distributions, dim=2))

class CombinedLoss(nn.Module):
    """Combined loss function that applies multiple losses with configurable weights."""
    
    def __init__(self, loss_weights: Dict[str, float]):
        super().__init__()
        self.loss_weights = loss_weights
    
    def forward(self, logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate combined loss from component losses."""
        total_loss = 0
        loss_values = {}
        
        # Apply each loss function if weight > 0
        loss_functions = {
            'mse': (mse_loss, 'mse_target'),
            'ce': (cross_entropy_loss, 'ce_target'),
            'soft_ce': (soft_cross_entropy_loss, 'soft_ce_target'),
            'kl': (kl_divergence_loss, 'kl_target')
        }
        
        for loss_name, (loss_fn, target_key) in loss_functions.items():
            weight = self.loss_weights.get(loss_name, 0)
            if weight > 0:
                loss_val = loss_fn(logits, batch[target_key])
                total_loss += weight * loss_val
                loss_values[loss_name] = loss_val.item()
        
        return total_loss, loss_values

# ============================================================================
# Training and Evaluation
# ============================================================================

def train_epoch(model: nn.Module, train_loader: DataLoader, optimizer, 
                loss_fn: CombinedLoss, device: torch.device, 
                scheduler=None, clip_grad_norm: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {}
    
    for batch in tqdm(train_loader, desc="Training"):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss, loss_values = loss_fn(logits, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        # Track losses
        total_loss += loss.item()
        for name, value in loss_values.items():
            loss_components[name] = loss_components.get(name, 0) + value
    
    # Average the losses
    avg_loss = total_loss / len(train_loader)
    avg_components = {name: value / len(train_loader) for name, value in loss_components.items()}
    
    return avg_loss, avg_components

def evaluate(model: nn.Module, val_loader: DataLoader, loss_fn: CombinedLoss, 
             device: torch.device, bin_size: float = 1/6) -> Tuple[float, Dict[str, float], float, float, float, float]:
    """Evaluate model with comprehensive metrics."""
    model.eval()
    total_loss = 0
    loss_components = {}
    all_logits = []
    all_mse_targets = []
    
    # Collect predictions and targets
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss, loss_values = loss_fn(logits, batch)
            
            total_loss += loss.item()
            for name, value in loss_values.items():
                loss_components[name] = loss_components.get(name, 0) + value
            
            all_logits.append(logits.cpu())
            all_mse_targets.append(batch['mse_target'])
    
    # Compute metrics
    metrics = _compute_evaluation_metrics(all_logits, all_mse_targets, bin_size)
    
    # Average loss
    avg_loss = total_loss / len(val_loader)
    avg_components = {name: value / len(val_loader) for name, value in loss_components.items()}
    
    # Log results
    _log_evaluation_results(metrics, avg_loss, avg_components)
    
    return avg_loss, avg_components, metrics['avg_correlation'], metrics['avg_binned_accuracy'], metrics['avg_mse'], metrics['avg_mae']

def _compute_evaluation_metrics(all_logits: List[torch.Tensor], all_mse_targets: List[torch.Tensor], 
                              bin_size: float) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics."""
    # Concatenate all predictions and targets
    all_logits = torch.cat(all_logits, dim=0)
    all_mse_targets = torch.cat(all_mse_targets, dim=0)
    
    # Compute expected scores from logits
    probs = F.softmax(all_logits, dim=2)
    possible_values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float)
    expected_scores = (probs * possible_values.view(1, 1, 5)).sum(dim=2)
    
    # Compute per-dimension metrics
    correlations = []
    binned_accuracies = []
    mses = []
    maes = []
    
    for dim in range(len(DIMENSIONS)):
        pred = expected_scores[:, dim].numpy()
        target = all_mse_targets[:, dim].numpy()
        
        # Correlation (handle edge cases)
        if len(np.unique(pred)) > 1 and len(np.unique(target)) > 1:
            corr, _ = pearsonr(pred, target)
        else:
            corr = 0.0
            logger.warning(f"Correlation for {DIMENSIONS[dim]} set to 0 due to constant values.")
        
        correlations.append(corr)
        
        # Regression metrics
        mse = np.mean((pred - target) ** 2)
        mae = np.mean(np.abs(pred - target))
        mses.append(mse)
        maes.append(mae)
        
        # Binned accuracy
        bin_acc = np.mean(np.abs(pred - target) < bin_size)
        binned_accuracies.append(bin_acc)
    
    return {
        'correlations': correlations,
        'binned_accuracies': binned_accuracies,
        'mses': mses,
        'maes': maes,
        'avg_correlation': np.mean(correlations),
        'avg_binned_accuracy': np.mean(binned_accuracies),
        'avg_mse': np.mean(mses),
        'avg_mae': np.mean(maes)
    }

def _log_evaluation_results(metrics: Dict, avg_loss: float, avg_components: Dict[str, float]):
    """Log comprehensive evaluation results."""
    logger.info("Evaluation Metrics:")
    logger.info(f"  Avg Pearson Correlation: {metrics['avg_correlation']:.4f}")
    logger.info(f"  Avg Binned Accuracy: {metrics['avg_binned_accuracy']:.4f}")
    logger.info(f"  Avg MSE: {metrics['avg_mse']:.4f}")
    logger.info(f"  Avg MAE: {metrics['avg_mae']:.4f}")
    
    for dim in range(len(DIMENSIONS)):
        logger.info(f"  {DIMENSIONS[dim]}: Corr {metrics['correlations'][dim]:.4f}, "
                   f"Bin Acc {metrics['binned_accuracies'][dim]:.4f}, "
                   f"MSE {metrics['mses'][dim]:.4f}, MAE {metrics['maes'][dim]:.4f}")
    
    logger.info(f"Evaluation Loss: {avg_loss:.4f}")
    logger.info(f"Loss Components: {', '.join([f'{k}: {v:.4f}' for k, v in avg_components.items()])}")

# ============================================================================
# Model Creation and Training
# ============================================================================

def create_model(config: Dict, tokenizer) -> BaseReadabilityModel:
    """Create the appropriate model based on configuration."""
    model_type = config.get("model_type", "improved")
    
    if model_type == "standard":
        logger.info("Using standard ReadabilityModel architecture")
        return ReadabilityModel(
            config["model_name"],
            tokenizer=tokenizer,
            num_dimensions=len(DIMENSIONS),
            num_scores=5,
            dropout_rate=config["dropout_rate"]
        )
    
    if model_type == "improved":
        logger.info(f"Using ImprovedReadabilityModel architecture with attention heads={config['attention_heads']}")
        return ImprovedReadabilityModel(
            config["model_name"],
            tokenizer=tokenizer,
            num_dimensions=len(DIMENSIONS),
            num_scores=5,
            attention_heads=config["attention_heads"],
            dropout_rate=config["dropout_rate"]
        )
    
    # Default to improved model for unknown types
    logger.warning(f"Unknown model type '{model_type}', defaulting to ImprovedReadabilityModel")
    return ImprovedReadabilityModel(
        config["model_name"],
        tokenizer=tokenizer,
        num_dimensions=len(DIMENSIONS),
        num_scores=5,
        attention_heads=config["attention_heads"],
        dropout_rate=config["dropout_rate"]
    )

def _create_optimizer_and_scheduler(model: nn.Module, train_loader: DataLoader, 
                                  config: Dict) -> Tuple[torch.optim.Optimizer, Any]:
    """Create optimizer and learning rate scheduler."""
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler

def _initialize_training_history() -> Dict[str, List]:
    """Initialize training history tracking dictionary."""
    return {
        'train_loss': [],
        'val_loss': [],
        'train_components': [],
        'val_components': [],
        'val_correlation': [],
        'val_binned_accuracy': [],
        'val_mse': [],
        'val_mae': []
    }

def _save_model_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                          epoch: int, metrics: Dict, config: Dict, 
                          history: Dict, filename: str):
    """Save model checkpoint with all relevant information."""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'history': history,
        **metrics  # Unpack all metrics
    }
    
    torch.save(checkpoint, os.path.join(config['output_dir'], filename))

def _check_early_stopping(val_loss: float, best_val_loss: float, patience_counter: int, 
                         patience: int) -> Tuple[bool, int, bool]:
    """Check if early stopping should be triggered and update counters."""
    improved = val_loss < best_val_loss
    
    if improved:
        return True, 0, False  # improved=True, reset_counter=0, should_stop=False
    
    new_counter = patience_counter + 1
    should_stop = new_counter >= patience
    
    return False, new_counter, should_stop

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, config: Dict) -> Tuple[nn.Module, Dict]:
    """Train the model with comprehensive tracking and early stopping."""
    # Setup training components
    loss_fn = CombinedLoss(config['loss_weights'])
    optimizer, scheduler = _create_optimizer_and_scheduler(model, train_loader, config)
    history = _initialize_training_history()
    
    # Early stopping setup
    best_val_loss = float('inf')
    patience = config.get('patience', 3)
    patience_counter = 0
    best_epoch = -1
    
    model.to(device)
    
    for epoch in range(config['epochs']):
        # Training phase
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, loss_fn, device, 
            scheduler, config['clip_grad_norm']
        )
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss, val_components, val_correlation, val_binned_accuracy, val_mse, val_mae = val_metrics
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_components'].append(train_components)
        history['val_components'].append(val_components)
        history['val_correlation'].append(val_correlation)
        history['val_binned_accuracy'].append(val_binned_accuracy)
        history['val_mse'].append(val_mse)
        history['val_mae'].append(val_mae)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics - Corr: {val_correlation:.4f}, Acc: {val_binned_accuracy:.4f}, MSE: {val_mse:.4f}, MAE: {val_mae:.4f}")
        logger.info(f"Train components: {', '.join([f'{k}: {v:.4f}' for k, v in train_components.items()])}")
        logger.info(f"Val components: {', '.join([f'{k}: {v:.4f}' for k, v in val_components.items()])}")
        
        # Check for improvement and handle early stopping
        improved, patience_counter, should_stop = _check_early_stopping(
            val_loss, best_val_loss, patience_counter, patience
        )
        
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            
            # Save best model
            if config['save_best']:
                metrics = {
                    'val_loss': val_loss,
                    'val_components': val_components,
                    'val_correlation': val_correlation,
                    'val_binned_accuracy': val_binned_accuracy,
                    'val_mse': val_mse,
                    'val_mae': val_mae
                }
                _save_model_checkpoint(model, optimizer, epoch, metrics, config, history, 'best_model.pt')
                logger.info(f"New best model saved with val loss: {best_val_loss:.4f}")
        else:
            logger.info(f"No improvement for {patience_counter} epochs. Best val loss: {best_val_loss:.4f}")
            
            if should_stop:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    final_metrics = {
        'val_loss': val_loss,
        'val_components': val_components,
        'val_correlation': val_correlation,
        'val_binned_accuracy': val_binned_accuracy,
        'val_mse': val_mse,
        'val_mae': val_mae
    }
    _save_model_checkpoint(model, optimizer, epoch, final_metrics, config, history, 'final_model.pt')
    
    # Save additional files
    torch.save(history, os.path.join(config['output_dir'], 'training_history.pt'))
    save_json(config, os.path.join(config['output_dir'], 'config.json'))
    logger.info(f"Config saved to {os.path.join(config['output_dir'], 'config.json')}")
    
    # Load best model if available
    if config['save_best']:
        checkpoint = torch.load(os.path.join(config['output_dir'], 'best_model.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {best_epoch+1} with val loss: {best_val_loss:.4f}")
    
    return model, history

def run_prediction_tests(model: BaseReadabilityModel, tokenizer, device: torch.device, config: Dict) -> List[Dict]:
    """Run prediction tests on the trained model."""
    logger.info("Testing inference...")
    
    example_text = "The cat sat on the mat. It was a sunny day."
    example_texts = [
        "The cat sat on the mat. It was a sunny day.",
        "The quantum chromodynamics framework elucidates how quarks interact through gluons."
    ]
    
    # Test single prediction
    result = model.predict_single(example_text)
    logger.info("Single text prediction:")
    logger.info(f"  Overall score: {result['overall_score']:.2f}")
    for dim, score in result['dimension_scores'].items():
        logger.info(f"  {dim}: {score:.2f}")
    
    # Test batch prediction
    results = model.predict_batch(example_texts)
    logger.info("\nBatch prediction:")
    for i, pred in enumerate(results):
        logger.info(f"Text {i+1} predictions:")
        logger.info(f"  Overall score: {pred['overall_score']:.2f}")
        for dim, score in pred['dimension_scores'].items():
            logger.info(f"  {dim}: {score:.2f}")
    
    return results

# ============================================================================
# Argument Parsing and Configuration
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model configuration."""
    parser = argparse.ArgumentParser(description="Train a readability scoring model.")
    
    # Data settings
    parser.add_argument("--file-patterns", "--file_patterns", type=str, nargs="+", 
                        default=["/data/home/djbf/storage/bls/rq1/outputs/phase0/*/llm/5/*/readability_metrics.json"],
                        help="Glob patterns to find JSON files")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--max-length", "--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--val-split", "--val_split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-split", "--test_split", type=float, default=0.15, help="Test split ratio")
    parser.add_argument("--bin-size", "--bin_size", type=float, default=1/6, help="Bin size for accuracy calculation")

    # Model settings
    parser.add_argument("--model-name", "--model_name", type=str, default="kamalkraj/BioSimCSE-BioLinkBERT-BASE",
                        help="Pretrained model name")
    parser.add_argument("--model-type", "--model_type", type=str, choices=["standard", "improved"], 
                        default="standard", help="Model architecture type")
    parser.add_argument("--attention-heads", "--attention_heads", type=int, default=2, help="Number of attention heads")
    parser.add_argument("--dropout-rate", "--dropout_rate", type=float, default=0.2, help="Dropout rate")
    
    # Loss settings
    parser.add_argument("--mse-weight", "--mse_weight", type=float, default=0.0, help="Weight for MSE loss")
    parser.add_argument("--kl-weight", "--kl_weight", type=float, default=0.0, help="Weight for KL divergence loss")
    parser.add_argument("--ce-weight", "--ce_weight", type=float, default=0.0, help="Weight for cross-entropy loss")
    parser.add_argument("--soft-ce-weight", "--soft_ce_weight", type=float, default=0.0, help="Weight for soft cross-entropy loss")
    
    # Model filters
    parser.add_argument("--mse-models", "--mse_models", type=str, nargs="*", default=None,
                        help="Models to use for MSE loss")
    parser.add_argument("--kl-models", "--kl_models", type=str, nargs="*", default=None,
                        help="Models to use for KL loss")
    parser.add_argument("--ce-models", "--ce_models", type=str, nargs="*", default=None,
                        help="Models to use for CE loss")
    parser.add_argument("--soft-ce-models", "--soft_ce_models", type=str, nargs="*", default=None,
                        help="Models to use for soft CE loss")
    
    # Dataset settings
    parser.add_argument("--soft-dist-type", "--soft_dist_type", type=str, default="squared_distance", 
                        choices=["squared_distance", "histogram"], help="Type of soft distribution")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax")
    
    # Training settings
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--clip-grad-norm", "--clip_grad_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Output settings
    parser.add_argument("--output-dir", "--output_dir", type=str, default="outputs/llmify/", help="Output directory")
    parser.add_argument("--save-best", "--save_best", action="store_true", default=True, help="Save best model")
    
    return parser.parse_args()

def create_config_from_args(args: argparse.Namespace) -> Dict:
    """Create configuration dictionary from parsed arguments."""
    return {
        # Script settings
        "regression_type": "classification",
        
        # Data settings
        "file_patterns": args.file_patterns,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "bin_size": args.bin_size,

        # Model settings
        "model_name": args.model_name,
        "model_type": args.model_type,
        "attention_heads": args.attention_heads,
        "dropout_rate": args.dropout_rate,
        
        # Loss settings
        "loss_weights": {
            "mse": args.mse_weight,
            "kl": args.kl_weight,
            "ce": args.ce_weight,
            "soft_ce": args.soft_ce_weight
        },
        
        # Model filters for different loss functions
        "model_filters": {
            "mse": args.mse_models,
            "kl": args.kl_models,
            "ce": args.ce_models,
            "soft_ce": args.soft_ce_models
        },
        
        # Dataset settings
        "soft_dist_type": args.soft_dist_type,
        "temperature": args.temperature,
        
        # Training settings
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "clip_grad_norm": args.clip_grad_norm,
        "patience": args.patience,
        "seed": args.seed,
        
        # Output settings
        "output_dir": args.output_dir,
        "save_best": args.save_best
    }

def setup_data_splits(samples: List[Dict], config: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, validation, and test sets."""
    # Split data into train+val and test
    train_val_samples, test_samples = train_test_split(
        samples, 
        test_size=config["test_split"], 
        random_state=config["seed"]
    )
    
    # Split train+val into train and val
    val_relative_split = config["val_split"] / (1 - config["test_split"])
    train_samples, val_samples = train_test_split(
        train_val_samples, 
        test_size=val_relative_split, 
        random_state=config["seed"]
    )
    
    logger.info(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")
    return train_samples, val_samples, test_samples

def create_datasets(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict], 
                   tokenizer, config: Dict) -> Tuple[ReadabilityDataset, ReadabilityDataset, ReadabilityDataset]:
    """Create PyTorch datasets for training, validation, and testing."""
    dataset_kwargs = {
        "tokenizer": tokenizer,
        "max_length": config["max_length"],
        "model_filters": config["model_filters"],
        "soft_dist_type": config["soft_dist_type"],
        "temperature": config["temperature"]
    }
    
    train_dataset = ReadabilityDataset(train_samples, **dataset_kwargs)
    val_dataset = ReadabilityDataset(val_samples, **dataset_kwargs)
    test_dataset = ReadabilityDataset(test_samples, **dataset_kwargs)
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset: ReadabilityDataset, val_dataset: ReadabilityDataset, 
                       test_dataset: ReadabilityDataset, config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch data loaders."""
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])
    
    return train_loader, val_loader, test_loader

def evaluate_and_save_test_results(model: nn.Module, test_loader: DataLoader, config: Dict, device: torch.device):
    """Evaluate model on test set and save results."""
    logger.info("Evaluating on test set...")
    
    loss_fn = CombinedLoss(config['loss_weights'])
    test_metrics = evaluate(model, test_loader, loss_fn, device, config['bin_size'])
    test_loss, test_components, test_correlation, test_binned_accuracy, test_mse, test_mae = test_metrics
    
    logger.info(f"Test Metrics - Loss: {test_loss:.4f}, Corr: {test_correlation:.4f}, "
               f"Bin Acc: {test_binned_accuracy:.4f}, MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")

    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_components': test_components,
        'test_correlation': test_correlation,
        'test_binned_accuracy': test_binned_accuracy,
        'test_mse': test_mse,
        'test_mae': test_mae
    }
    
    torch.save(test_results, os.path.join(config['output_dir'], 'test_results.pt'))
    save_json(test_results, os.path.join(config['output_dir'], 'test_results.json'))
    logger.info(f"Test results saved to {os.path.join(config['output_dir'], 'test_results.pt')}")

def create_training_plots(history: Dict, config: Dict):
    """Create and save training progress plots."""
    import matplotlib.pyplot as plt
    
    plots_dir = os.path.join(config["output_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot overall loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'overall_loss.png'))
    plt.close()
    
    # Plot component losses
    plt.figure(figsize=(12, 8))
    for loss_type in history['train_components'][0].keys():
        train_values = [comp[loss_type] for comp in history['train_components']]
        val_values = [comp[loss_type] for comp in history['val_components']]
        plt.plot(train_values, label=f'Train {loss_type}')
        plt.plot(val_values, label=f'Val {loss_type}')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'component_losses.png'))
    plt.close()
    
    # Plot additional metrics
    plt.figure(figsize=(12, 8))
    plt.plot(history['val_correlation'], label='Val Correlation')
    plt.plot(history['val_binned_accuracy'], label='Val Binned Accuracy')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.title('Validation Metrics Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, 'validation_metrics.png'))
    plt.close()
    
    logger.info(f"Saved loss and metric plots to {plots_dir}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to train and test the readability model."""
    # Parse arguments and create configuration
    args = parse_args()
    config = create_config_from_args(args)

    # Ensure at least one loss weight is positive
    if sum(config["loss_weights"].values()) <= 0:
        raise ValueError("At least one loss weight must be positive.")
    
    # Setup device and random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # Load and split data
    logger.info("Loading data...")
    samples = load_data(config["file_patterns"])
    train_samples, val_samples, test_samples = setup_data_splits(samples, config)
    
    # Create tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_samples, val_samples, test_samples, tokenizer, config
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, config
    )
    
    # Create and train model
    model = create_model(config, tokenizer)
    logger.info("Training model...")
    trained_model, history = train_model(model, train_loader, val_loader, device, config)
    
    # Evaluate on test set
    evaluate_and_save_test_results(trained_model, test_loader, config, device)
    
    # Create training plots
    create_training_plots(history, config)
    
    # Test inference
    run_prediction_tests(trained_model, tokenizer, device, config)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()