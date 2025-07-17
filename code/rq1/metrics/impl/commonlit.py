import os
import gc
import logging
from glob import glob
from typing import List, Union, Dict, Any
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

from utils.helpers import setup_logging

logger = setup_logging()

MAX_LENGTH = 512

# Inspired by the REference-FREE Model-Based Metric for Text Simplification
class TransformerReadabilityClassifier:
    """Classifier for readability prediction using any transformer model."""
    
    def __init__(self, model_path: str, model_name: str = None):
        """Initialize with a pre-trained model."""
        self.device = self._get_device()
        self.model_name = model_name or os.path.basename(model_path)
        
        logger.info(f"Initializing {self.model_name} on {self.device}")
        
        # Load model components
        self.config = self._load_config(model_path)
        self.model = self._load_model(model_path)
        self.tokenizer = self._load_tokenizer(model_path)
        
        # Setup model for inference
        self._setup_model_for_inference()
        self._warmup()
        
        logger.info(f"Model {self.model_name} initialization complete")
    
    def _get_device(self) -> str:
        """Determine the appropriate device for inference."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_config(self, model_path: str) -> AutoConfig:
        """Load model configuration."""
        logger.debug(f"Loading config from {model_path}")
        return AutoConfig.from_pretrained(model_path, num_labels=1)
    
    def _load_model(self, model_path: str) -> AutoModelForSequenceClassification:
        """Load the transformer model."""
        logger.debug(f"Loading model weights from {model_path}")
        return AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
    
    def _load_tokenizer(self, model_path: str) -> AutoTokenizer:
        """Load the tokenizer."""
        logger.debug(f"Loading tokenizer from {model_path}")
        return AutoTokenizer.from_pretrained(model_path)
    
    def _setup_model_for_inference(self) -> None:
        """Configure model for inference mode."""
        self.model.to(self.device)
        logger.info(f"Model {self.model_name} moved to {self.device}")
        
        self.model.eval()
        
        # Convert to float for CPU usage
        if self.device == "cpu":
            logger.info(f"Converting model {self.model_name} to float for CPU usage")
            self.model.float()
    
    def _warmup(self) -> None:
        """Warm up the model with a dummy input."""
        logger.debug(f"Warming up model {self.model_name}")
        
        dummy_input = self.tokenizer(
            "Warm up text",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            self.model(**dummy_input)
        
        logger.debug(f"Model {self.model_name} warm-up complete")
    
    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def _process_single_prediction(self, logits: torch.Tensor) -> float:
        """Process logits for a single prediction."""
        result = float(logits)
        logger.debug(f"Single prediction: {result}")
        return result
    
    def _process_batch_predictions(self, logits: torch.Tensor) -> List[float]:
        """Process logits for batch predictions."""
        results = logits.tolist()
        logger.debug(f"Batch predictions: min={min(logits)}, max={max(logits)}, mean={sum(logits)/len(logits)}")
        return results
    
    def _clear_cuda_cache_if_needed(self) -> None:
        """Clear CUDA cache if using GPU."""
        if self.device.startswith('cuda'):
            logger.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()
    
    def _process_batch_logits(self, logits: torch.Tensor, batch_size: int) -> List[float]:
        """Process model logits based on batch size."""
        cpu_logits = logits.cpu().numpy().squeeze()
        
        if batch_size == 1:
            return [self._process_single_prediction(cpu_logits)]
        
        return self._process_batch_predictions(cpu_logits)
    
    @torch.no_grad()
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Predict readability scores for a batch of texts."""
        if not texts:
            logger.warning("Empty text list provided to predict_batch")
            return []
        
        logger.info(f"Predicting readability scores for {len(texts)} texts with model {self.model_name}")
        logger.debug(f"Using batch size: {batch_size}")
        
        predictions = []
        batches = list(range(0, len(texts), batch_size))
        
        for i in tqdm(batches, desc=f"Predicting with {self.model_name}"):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.debug(f"Processing batch {batch_num}/{len(batches)}, size: {len(batch)}")
            
            # Tokenize and predict
            inputs = self._tokenize_batch(batch)
            outputs = self.model(**inputs)
            
            # Process results
            batch_predictions = self._process_batch_logits(outputs.logits, len(batch))
            predictions.extend(batch_predictions)
            
            # Clean up GPU memory
            self._clear_cuda_cache_if_needed()
        
        logger.info(f"Finished predictions with model {self.model_name}")
        return predictions
    
    def predict_single(self, text: str) -> float:
        """Predict readability score for a single text."""
        logger.info(f"Predicting readability score for a single text with model {self.model_name}")
        return self.predict_batch([text])[0]
    
    def unload(self) -> None:
        """Unload model from GPU to free memory."""
        logger.info(f"Unloading model {self.model_name} from {self.device}")
        
        # Move to CPU and delete components
        self.model = self.model.cpu()
        logger.debug("Deleting model components to free memory")
        
        del self.model
        del self.tokenizer
        del self.config
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            logger.debug("Clearing CUDA cache")
            torch.cuda.empty_cache()
        
        # Force garbage collection
        logger.debug("Running garbage collection")
        gc.collect()
        
        logger.info(f"Model {self.model_name} successfully unloaded")


class DebertaReadabilityEnsemble:
    """Ensemble of Deberta models for readability prediction."""
    
    def __init__(self, model_dir: str):
        """Initialize with models from a directory."""
        logger.info(f"Initializing DeBERTa Ensemble from directory: {model_dir}")
        
        self.model_paths = self._get_model_paths(model_dir)
        self.fold_names = self._generate_fold_names()
        
        logger.info(f"Found {len(self.model_paths)} models for ensemble: {', '.join(self.fold_names)}")
    
    def _get_model_paths(self, model_dir: str) -> List[str]:
        """Get sorted list of model paths from directory."""
        model_paths = sorted(glob(os.path.join(model_dir, "model_fold_*")))
        
        if not model_paths:
            logger.error(f"No models found in {model_dir}")
            raise ValueError(f"No models could be loaded from {model_dir}")
        
        return model_paths
    
    def _generate_fold_names(self) -> List[str]:
        """Generate fold names for ensemble models."""
        return [f"fold_{i+1}" for i in range(len(self.model_paths))]
    
    def _load_fold_model(self, fold_path: str, fold_name: str) -> TransformerReadabilityClassifier:
        """Load a single fold model."""
        logger.info(f"Loading model {fold_name} from {fold_path}")
        best_path = os.path.join(fold_path, "best")
        return TransformerReadabilityClassifier(best_path, model_name=fold_name)
    
    def _unload_fold_model(self, model: TransformerReadabilityClassifier, fold_name: str) -> None:
        """Unload a fold model and clean up memory."""
        logger.info(f"Unloading model {fold_name}")
        model.unload()
        del model
        torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"Model {fold_name} processed and unloaded successfully")
    
    def _calculate_ensemble_score(self, fold_scores: Dict[str, float], text_idx: int) -> float:
        """Calculate ensemble score from individual fold scores."""
        valid_scores = [score for score in fold_scores.values() if isinstance(score, (int, float))]
        
        if not valid_scores:
            logger.warning(f"No valid scores for text at index {text_idx}, using default 0.0")
            return 0.0
        
        mean_score = sum(valid_scores) / len(valid_scores)
        logger.debug(f"Text {text_idx}: Score={mean_score:.4f} (from {len(valid_scores)} models)")
        return mean_score
    
    def _format_ensemble_result(self, fold_scores: Dict[str, float], ensemble_score: float) -> Dict[str, float]:
        """Format ensemble result with individual fold scores."""
        result = {"ensemble": ensemble_score}
        result.update(fold_scores)
        return result
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, float]]:
        """Predict readability scores using the ensemble."""
        if not texts:
            logger.warning("Empty text list provided to ensemble predict_batch")
            return []
        
        logger.info(f"Predicting readability scores for {len(texts)} texts with DeBERTa ensemble")
        all_predictions = [{} for _ in range(len(texts))]
        
        # Process each fold model
        for i, fold_path in enumerate(tqdm(self.model_paths, desc="Processing model folds")):
            fold_name = self.fold_names[i]
            logger.info(f"Processing ensemble member {i+1}/{len(self.model_paths)}: {fold_name}")
            
            # Load model, predict, and unload
            model = self._load_fold_model(fold_path, fold_name)
            predictions = model.predict_batch(texts, batch_size)
            logger.info(f"Obtained predictions from model {fold_name}")
            
            # Store predictions
            for text_idx, pred in enumerate(predictions):
                all_predictions[text_idx][fold_name] = pred
            
            # Clean up model
            self._unload_fold_model(model, fold_name)
        
        # Calculate ensemble results
        logger.info("Calculating ensemble scores")
        results = []
        
        for text_idx, fold_scores in enumerate(all_predictions):
            ensemble_score = self._calculate_ensemble_score(fold_scores, text_idx)
            result = self._format_ensemble_result(fold_scores, ensemble_score)
            results.append(result)
        
        logger.info(f"Ensemble prediction complete for {len(texts)} texts")
        return results
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """Predict readability score for a single text using the ensemble."""
        logger.info("Predicting readability score for a single text with DeBERTa ensemble")
        return self.predict_batch([text])[0]


class CommonLitClassifier:
    """Classifier for readability prediction using Deberta or Albert models."""
    
    def __init__(self, model_type: str, model_path: str, fold_name: str = None):
        """Initialize with the specified model type and path."""
        logger.info(f"Initializing CommonLitClassifier with model_type={model_type}")
        logger.info(f"Model path: {model_path}")
        
        self.model_type = model_type
        self.model = self._create_model(model_type, model_path, fold_name)
        self.is_ensemble = model_type == 'deberta_ensemble'
        
        logger.info("CommonLitClassifier initialization complete")
    
    def _create_model(self, model_type: str, model_path: str, fold_name: str) -> Union[DebertaReadabilityEnsemble, TransformerReadabilityClassifier]:
        """Create the appropriate model based on model type."""
        if model_type == 'deberta_ensemble':
            return self._create_ensemble_model(model_path)
        
        if model_type in ['deberta_single', 'albert']:
            return self._create_single_model(model_type, model_path, fold_name)
        
        logger.error(f"Invalid model_type: {model_type}")
        raise ValueError("Invalid model_type")
    
    def _create_ensemble_model(self, model_path: str) -> DebertaReadabilityEnsemble:
        """Create ensemble model."""
        logger.info("Using DeBERTa ensemble model")
        return DebertaReadabilityEnsemble(model_path)
    
    def _create_single_model(self, model_type: str, model_path: str, fold_name: str) -> TransformerReadabilityClassifier:
        """Create single transformer model."""
        resolved_fold_name = self._resolve_fold_name(model_type, model_path, fold_name)
        logger.info(f"Using fold name: {resolved_fold_name}")
        return TransformerReadabilityClassifier(model_path, model_name=resolved_fold_name)
    
    def _resolve_fold_name(self, model_type: str, model_path: str, fold_name: str) -> str:
        """Resolve fold name based on model type and path."""
        if fold_name is not None:
            return fold_name
        
        if model_type == 'albert':
            logger.info("Using ALBERT model")
            return 'albert'
        
        # deberta_single case
        logger.info("Using single DeBERTa model")
        return os.path.basename(os.path.dirname(model_path))
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> Union[List[Dict[str, float]], List[float]]:
        """Predict readability scores for a batch of texts."""
        logger.info(f"CommonLitClassifier predicting for {len(texts)} texts (is_ensemble={self.is_ensemble})")
        
        if self.is_ensemble:
            logger.info("Using ensemble prediction")
            return self.model.predict_batch(texts, batch_size)
        
        logger.info("Using single model prediction")
        return self.model.predict_batch(texts, batch_size)
    
    def predict_single(self, text: str) -> Union[Dict[str, float], float]:
        """Predict readability score for a single text."""
        logger.info("CommonLitClassifier predicting for a single text")
        return self.predict_batch([text])[0]
    
    def unload(self) -> None:
        """Unload model from GPU to free memory if applicable."""
        logger.info("Unloading CommonLitClassifier")
        
        if not self.is_ensemble:
            logger.info("Unloading single model")
            self.model.unload()
            return
        
        logger.info("No need to explicitly unload ensemble (individual models already unloaded)")
        logger.info("CommonLitClassifier unloaded successfully")