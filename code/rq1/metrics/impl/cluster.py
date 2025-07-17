import os
import joblib
import gc
from typing import List, Dict, Tuple, Union, Optional
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import fasttext
import spacy
from datasets import load_dataset

from utils.helpers import setup_logging

logger = setup_logging()

# Xgboost overfitting...

class ClearCorpusLoader:
    """Loader for the CLEAR corpus with standardized column names."""
    
    @staticmethod
    def load() -> Tuple[List[str], List[float], List[str], List[float]]:
        """Load the CLEAR corpus, returning train and validation splits."""
        logger.info("Loading CLEAR corpus (train and validation splits)...")
        
        # Load train split
        train_dataset = load_dataset("casey-martin/CommonLit-Ease-of-Readability", split="train")
        train_df = train_dataset.to_pandas()
        train_df = train_df.rename(columns={'Excerpt': 'text', 'BT_easiness': 'readability_score'})
        train_texts = train_df['text'].tolist()
        train_scores = train_df['readability_score'].astype(float).tolist()
        
        # Load validation split
        eval_dataset = load_dataset("casey-martin/CommonLit-Ease-of-Readability", split="validation")
        eval_df = eval_dataset.to_pandas()
        eval_df = eval_df.rename(columns={'Excerpt': 'text', 'BT_easiness': 'readability_score'})
        eval_texts = eval_df['text'].tolist()
        eval_scores = eval_df['readability_score'].astype(float).tolist()
        
        logger.info(f"Loaded {len(train_texts)} train and {len(eval_texts)} validation samples from CLEAR corpus")
        return train_texts, train_scores, eval_texts, eval_scores


class MedReadmeCorpusLoader:
    """Loader for the MEDREADME corpus with standardized column names."""
    
    @staticmethod
    def load(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[float], List[str], List[float]]:
        """Load the MEDREADME corpus and create train-test split."""
        logger.info("Loading MEDREADME corpus and creating train-test split...")
        
        # Assume data_path points directly to readability.csv
        df = pd.read_csv(data_path)
        
        # Standardize column names
        df = df.rename(columns={'Sentence': 'text', 'Readability': 'readability_score'})
        
        # Extract texts and scores
        texts = df['text'].tolist()
        scores = df['readability_score'].astype(float).tolist()
        
        # Create train-test split
        train_texts, eval_texts, train_scores, eval_scores = train_test_split(
            texts, scores, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Created {len(train_texts)} train and {len(eval_texts)} test samples from MEDREADME corpus")
        return train_texts, train_scores, eval_texts, eval_scores


class ClusterModel:
    """Container for the trained clustering and regression models."""
    
    def __init__(self, kmeans_model, regression_model, n_clusters):
        """Initialize with trained models."""
        self.kmeans = kmeans_model
        self.regression_model = regression_model
        self.n_clusters = n_clusters


class ClusterBasedReadabilityTrainer:
    """
    Trainer for the cluster-based readability classifier from the paper:
    Language Modeling by Clustering with Word Embeddings for Text Readability Assessment
    Miriam Cha, Youngjune Gwon, H.T. Kung
    """
    
    def __init__(
        self,
        word_embedding_path: str,
        n_clusters: int = 100,
        model_name: str = "en_core_web_trf",
        random_state: int = 42
    ):
        """Initialize the trainer with embedding path and parameters."""
        self.word_embedding_path = word_embedding_path
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # Load fastText model
        logger.info(f"Loading fastText model from {word_embedding_path}")
        self.ft_model = self._load_word_embeddings(word_embedding_path)
        
        # Get embedding dimension
        self.embedding_dim = len(self.get_word_vector("the"))
        logger.info(f"Word embedding dimension: {self.embedding_dim}")
        
        # Load spaCy model
        logger.info(f"Loading spaCy model: {model_name}")
        spacy.prefer_gpu()  # Use GPU if available

        self.nlp = spacy.load(model_name, disable=["ner", "parser"])
        
        # Initialize clustering model
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto"
        )
        
        # Initialize regression model
        self.regression_model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='linear'))
        ])

        # self.regression_model = Pipeline([
        #     ('scaler', StandardScaler()),
        #     ('xgb', XGBRegressor(
        #         n_estimators=300,
        #         learning_rate=0.1,
        #         max_depth=7,
        #         objective='reg:squarederror',
        #         random_state=self.random_state
        #     ))
        # ])

        spacy.prefer_gpu()  # Use GPU if available
    
    def _load_word_embeddings(self, path: str):
        """Load fastText word embeddings."""
        return fasttext.load_model(path)
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get embedding vector for a word."""
        return self.ft_model.get_word_vector(word)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Extract tokens from text."""
        doc = self.nlp(text)
        tokens = [
            token.text.lower() for token in doc 
            if not token.is_punct and not token.is_space
        ]
        return tokens
    
    def _extract_vectors(self, tokens: List[str]) -> np.ndarray:
        """Get word vectors for tokens."""
        if not tokens:
            return np.zeros((0, self.embedding_dim))
            
        vectors = []
        for token in tokens:
            vectors.append(self.get_word_vector(token))
        
        return np.array(vectors)
    
    def _collect_word_vectors(self, texts: List[str]) -> np.ndarray:
        """Extract all word vectors from texts for clustering."""
        all_vectors = []
        
        for text in tqdm(texts, desc="Extracting word vectors"):
            tokens = self._preprocess_text(text)
            vectors = self._extract_vectors(tokens)
            if vectors.shape[0] > 0:
                all_vectors.append(vectors)
        
        if not all_vectors:
            raise ValueError("No word vectors extracted from texts")
            
        return np.vstack(all_vectors)
    
    def _create_histogram(self, text: str) -> np.ndarray:
        """Create cluster histogram for a text."""
        tokens = self._preprocess_text(text)
        if not tokens:
            return np.zeros(self.n_clusters)
            
        vectors = self._extract_vectors(tokens)
        if vectors.shape[0] == 0:
            return np.zeros(self.n_clusters)
            
        # Predict clusters
        clusters = self.kmeans.predict(vectors)
        
        # Create histogram
        histogram = np.zeros(self.n_clusters)
        for cluster in clusters:
            histogram[cluster] += 1
        
        # Normalize
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def train(self, texts: List[str], scores: List[float], test_size: float = 0.2) -> Dict:
        """Train models on the provided texts and scores."""
        if not texts or not scores:
            raise ValueError("Empty texts or scores provided")
            
        # Extract word vectors for clustering
        logger.info("Collecting word vectors for clustering...")
        word_vectors = self._collect_word_vectors(texts)
        
        # Fit K-means
        logger.info(f"Clustering with K-means ({self.n_clusters} clusters)...")
        self.kmeans.fit(word_vectors)
        
        # Create histograms
        logger.info("Creating cluster histograms...")
        histograms = np.array([self._create_histogram(text) for text in tqdm(texts, desc="Creating histograms")])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            histograms, scores, test_size=test_size, random_state=self.random_state
        )
        
        # Train regression model
        logger.info("Training regression model...")
        self.regression_model.fit(X_train, y_train)
        
        # Evaluate
        y_train_pred = self.regression_model.predict(X_train)
        y_test_pred = self.regression_model.predict(X_test)
        
        results = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_mse": mean_squared_error(y_test, y_test_pred),
            "test_r2": r2_score(y_test, y_test_pred),
            "n_train": len(X_train),
            "n_test": len(X_test)
        }
        
        logger.info(f"Training results: {results}")
        return results
    
    def evaluate(self, texts: List[str], scores: List[float]) -> Dict:
        """Evaluate trained models on the provided texts and scores."""
        if not hasattr(self, 'kmeans') or not hasattr(self, 'regression_model'):
            raise ValueError("Models must be trained first")
            
        # Create histograms
        histograms = np.array([self._create_histogram(text) for text in tqdm(texts, desc="Processing evaluation data")])
        
        # Make predictions
        predictions = self.regression_model.predict(histograms)
        
        # Calculate metrics
        results = {
            "mse": mean_squared_error(scores, predictions),
            "r2": r2_score(scores, predictions),
            "n_samples": len(texts),
            "pearson": np.corrcoef(scores, predictions)[0, 1]
        }
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def save_models(self, path: str) -> None:
        """Save trained models to files."""
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        kmeans_path = os.path.join(path, "kmeans.joblib")
        regression_path = os.path.join(path, "regression.joblib")
        
        joblib.dump(self.kmeans, kmeans_path)
        joblib.dump(self.regression_model, regression_path)
        
        metadata_path = os.path.join(path, "metadata.pkl")
        metadata = {
            "n_clusters": self.n_clusters,
            "embedding_dim": self.embedding_dim,
            "word_embedding_path": self.word_embedding_path
        }
        
        joblib.dump(metadata, metadata_path)
        logger.info(f"Models saved to {path}")
    
    def get_trained_model(self) -> ClusterModel:
        """Get the trained model package for the classifier."""
        if not hasattr(self, 'kmeans') or not hasattr(self, 'regression_model'):
            raise ValueError("Models must be trained first")
            
        return ClusterModel(
            kmeans_model=self.kmeans,
            regression_model=self.regression_model,
            n_clusters=self.n_clusters
        )


class ClusterBasedReadabilityClassifier:
    """Classifier for predicting text readability using trained models."""
    
    def __init__(
        self,
        word_embedding_path: str,
        model: Optional[ClusterModel] = None,
        model_name: str = "en_core_web_trf"
    ):
        """Initialize classifier with embedding path and optional model."""
        self.word_embedding_path = word_embedding_path
        
        # Load fastText model
        logger.info(f"Loading fastText model from {word_embedding_path}")
        self.ft_model = self._load_word_embeddings(word_embedding_path)
        
        # Load spaCy model
        logger.info(f"Loading spaCy model: {model_name}")
        self.nlp = spacy.load(model_name, disable=["ner", "parser"])
        
        # Set trained models if provided
        if model:
            self.kmeans = model.kmeans
            self.regression_model = model.regression_model
            self.n_clusters = model.n_clusters
        else:
            self.kmeans = None
            self.regression_model = None
            self.n_clusters = None
    
        spacy.prefer_gpu()  # Use GPU if available

    def _load_word_embeddings(self, path: str):
        """Load fastText word embeddings."""
        return fasttext.load_model(path)
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """Get embedding vector for a word."""
        return self.ft_model.get_word_vector(word)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Extract tokens from text."""
        doc = self.nlp(text)
        tokens = [
            token.text.lower() for token in doc 
            if not token.is_punct and not token.is_space
        ]
        return tokens
    
    def _extract_vectors(self, tokens: List[str]) -> np.ndarray:
        """Get word vectors for tokens."""
        if not tokens:
            return np.zeros((0, self.get_word_vector("the").shape[0]))
            
        vectors = []
        for token in tokens:
            vectors.append(self.get_word_vector(token))
        
        return np.array(vectors)
    
    def _create_histogram(self, text: str) -> np.ndarray:
        """Create cluster histogram for a text."""
        if not self.kmeans:
            raise ValueError("No trained model loaded")
            
        tokens = self._preprocess_text(text)
        if not tokens:
            return np.zeros(self.n_clusters)
            
        vectors = self._extract_vectors(tokens)
        if vectors.shape[0] == 0:
            return np.zeros(self.n_clusters)
            
        # Predict clusters
        clusters = self.kmeans.predict(vectors)
        
        # Create histogram
        histogram = np.zeros(self.n_clusters)
        for cluster in clusters:
            histogram[cluster] += 1
        
        # Normalize
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        return histogram
    
    def predict_single(self, text: str) -> float:
        """Predict readability score for a single text."""
        if not self.kmeans or not self.regression_model:
            raise ValueError("No trained model loaded")
        
        histogram = self._create_histogram(text)
        # Return score directly instead of dictionary
        return float(self.regression_model.predict([histogram])[0])
        
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """Predict readability scores for a batch of texts."""
        if not self.kmeans or not self.regression_model:
            raise ValueError("No trained model loaded")
        
        # Handle empty list
        if not texts:
            return []
        
        results = []
        
        # Process in batches to avoid memory issues with large inputs
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Create histograms for batch
            histograms = np.array([
                self._create_histogram(text) for text in tqdm(batch_texts, desc=f"Processing batch {i//batch_size + 1}")
            ])
            
            # Get predictions
            scores = self.regression_model.predict(histograms)
            
            # Return scores directly without wrapping in dictionaries
            batch_results = [float(score) for score in scores]
            results.extend(batch_results)
        
        return results
    
    def load_model(self, path: str) -> None:
        """Load trained models from files."""
        # Load metadata
        metadata_path = os.path.join(path, "metadata.pkl")
        metadata = joblib.load(metadata_path)
        
        # Load models
        kmeans_path = os.path.join(path, "kmeans.joblib")
        regression_path = os.path.join(path, "regression.joblib")
        
        self.kmeans = joblib.load(kmeans_path)
        self.regression_model = joblib.load(regression_path)
        
        # Set attributes
        self.n_clusters = metadata["n_clusters"]
        
        logger.info(f"Models loaded from {path}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get importance of each cluster in readability prediction."""
        if not self.regression_model:
            raise ValueError("No trained model loaded")
        
        # Extract feature importance from SVR
        importance = self.regression_model.named_steps['svr'].coef_
        
        # Ensure importance is 1-dimensional
        if importance.ndim > 1:
            importance = importance.ravel()
        
        # Create DataFrame
        df = pd.DataFrame({
            'cluster': np.arange(self.n_clusters),
            'importance': importance
        })
        
        # Sort by absolute importance
        df['abs_importance'] = np.abs(df['importance'])
        df = df.sort_values('abs_importance', ascending=False).reset_index(drop=True)
        
        return df


if __name__ == "__main__":
    # Configuration
    biowordvec_path = "/data/home/djbf/storage/bls/resources/models/BioWordVec/BioWordVec_PubMed_MIMICIII_d200.bin"
    model_path = "/data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/medreadme"
    medreadme_data_path = "/data/home/djbf/storage/bls/resources/datasets/medreadme/readability.csv"
    
    training = True
    dataset = "medreadme"

    # Load data
    if dataset == "medreadme":
        train_texts, train_scores, eval_texts, eval_scores = MedReadmeCorpusLoader.load(
            data_path=medreadme_data_path, test_size=0.2, random_state=42
        )
    else:
        train_texts, train_scores, eval_texts, eval_scores = ClearCorpusLoader.load()
    
    # Training or inference
    if training:
        # 1. Create trainer
        trainer = ClusterBasedReadabilityTrainer(
            word_embedding_path=biowordvec_path,
            n_clusters=300
        )
        
        # 2. Train models
        results = trainer.train(texts=train_texts, scores=train_scores)
        print(f"Training results: {results}")
        
        # 3. Evaluate on validation set
        eval_results = trainer.evaluate(texts=eval_texts, scores=eval_scores)
        print(f"Evaluation results: {eval_results}")
        
        # 4. Save models
        trainer.save_models(model_path)
        
        # 5. Get the trained model for immediate use
        trained_model = trainer.get_trained_model()

        # Delete trainer to free up memory
        del trainer
        gc.collect()
        
        # 6. Create classifier with trained model
        classifier = ClusterBasedReadabilityClassifier(
            word_embedding_path=biowordvec_path,
            model=trained_model
        )
    
    # Inference mode
    else:
        # 1. Create classifier
        classifier = ClusterBasedReadabilityClassifier(
            word_embedding_path=biowordvec_path
        )
        
        # 2. Load trained model
        classifier.load_model(model_path)
    
    # Make predictions
    example_texts = [
        "Furuncles and carbuncles, caused primarily by Staphylococcus aureus infection, demonstrate high curability with appropriate intervention. Treatment protocol typically involves incision and drainage for lesions >5mm, coupled with culture-guided antimicrobial therapy when indicated. MRSA consideration necessary in high-risk populations. Resolution typically occurs within 7-10 days with appropriate treatment. Recurrence rates approximately 10%, necessitating evaluation for predisposing factors including diabetes mellitus, immunosuppression, or chronic colonization. Preventive measures include proper hygiene and decolonization protocols for recurrent cases.",
        "Yes, boils and carbuncles are completely curable conditions. These painful skin infections usually clear up with proper treatment, which may include draining the infection and sometimes antibiotics. Most cases heal within 1-2 weeks with appropriate care. While they can come back, especially if there are underlying health issues, following good hygiene practices and completing any prescribed treatments helps prevent recurrence. If you get frequent boils, it's important to see a healthcare provider to check for any underlying conditions that might make you more susceptible.",
        "Metatarsalgia reversibility depends on causal factors and chronicity. Primary mechanical causes respond to conservative measures including offloading, orthotic devices, and appropriate footwear modification. Secondary causes require management of underlying conditions (Morton's neuroma, stress fractures, arthritis). Rehabilitation protocol includes intrinsic foot strengthening, gait retraining, and activity modification. Surgical intervention reserved for refractory cases or structural abnormalities.",
        "Yes, metatarsalgia (pain in the ball of your foot) can usually be improved or reversed, especially if treated early. Treatment typically includes wearing proper shoes, using shoe inserts or orthotics, resting, and exercises to strengthen your feet. The success of treatment depends on what's causing the pain. Most people see improvement with conservative treatment, though some might need more time or different approaches. Your podiatrist can create a specific treatment plan for your situation."
    ]
    
    predictions = classifier.predict_batch(example_texts)
    
    for text, score in zip(example_texts, predictions):
        print(f"Text: {text[:50]}...")
        print(f"Predicted readability: {score}")
        print()
    
    # Analyze feature importance
    importance = classifier.get_feature_importance()
    print("Top 10 most important clusters:")
    print(importance.head(10))