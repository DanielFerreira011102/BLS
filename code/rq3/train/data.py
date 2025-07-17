import random
import pandas as pd
import datasets
from torch.utils.data import Sampler
from typing import List, Iterator, Tuple
from utils.helpers import setup_logging

logger = setup_logging()

def load_complexity_data(data_path: str, config, test_mode: bool = False, test_sample_size: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Load and preprocess complexity data from CSV."""
    df = pd.read_csv(data_path)
    
    if test_mode:
        sample_size = min(test_sample_size, len(df))
        logger.info(f"Sampling {sample_size} rows for test mode")
        df = df.sample(n=sample_size, random_state=seed)
    
    # Get complexity column from config
    complexity_column = config.complexity.complexity_column
    
    # Check if the complexity column exists
    if complexity_column not in df.columns:
        available_columns = [col for col in df.columns if 'complexity' in col.lower()]
        logger.error(f"Complexity column '{complexity_column}' not found in data.")
        logger.info(f"Available complexity columns: {available_columns}")
        raise ValueError(f"Complexity column '{complexity_column}' not found. Available: {available_columns}")
    
    # Filter out rows with missing values in key columns
    required_cols = ["question", "text", complexity_column, "context_id"]
    original_len = len(df)
    df = df.dropna(subset=required_cols)
    
    if len(df) < original_len:
        logger.warning(f"Dropped {original_len - len(df)} rows with missing values")
    
    # Ensure complexity score is numeric
    df[complexity_column] = pd.to_numeric(df[complexity_column])
    
    # Log data summary
    logger.info(f"Loaded {len(df)} samples from {data_path}")
    logger.info(f"Found {len(df['context_id'].unique())} unique contexts")
    logger.info(f"Using complexity column: {complexity_column}")
    logger.info(f"Complexity values: {sorted(df[complexity_column].unique())}")
    
    return df

def prepare_training_data(df: pd.DataFrame, config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare training and validation datasets."""
    # Set random seed for reproducibility
    random.seed(config.seed)
    
    # Get complexity column from config
    complexity_column = config.complexity.complexity_column
    
    # Add complexity tokens if using control tokens
    if config.complexity.use_control_tokens:
        df["complexity_token"] = df[complexity_column].apply(
            lambda x: config.complexity.token_format.format(level=int(x))
        )
    
    # Split by context to avoid data leakage
    contexts = df["context_id"].unique().tolist()
    val_size = int(len(contexts) * config.data.validation_split)
    
    val_contexts = random.sample(contexts, k=val_size)
    train_df = df[~df["context_id"].isin(val_contexts)].copy()
    val_df = df[df["context_id"].isin(val_contexts)].copy()
    
    logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation samples")
    logger.info(f"Training contexts: {len(train_df['context_id'].unique())} / Validation: {len(val_df['context_id'].unique())}")
    
    return train_df, val_df

def create_dataset(df: pd.DataFrame, tokenizer, config):
    """Create dataset with appropriate complexity formatting."""
    formatted_data = []
    
    # Get complexity column from config
    complexity_column = config.complexity.complexity_column
    
    for _, row in df.iterrows():
        complexity_level = int(row[complexity_column])
        question = row["question"]
        answer = row["text"]
        
        # Format prompt based on configuration
        if config.complexity.use_control_tokens:
            # Use complexity tokens
            complexity_token = config.complexity.token_format.format(level=complexity_level)
            user_content = f"{complexity_token} {question}"
        else:
            # Use natural language prompts
            user_content = config.complexity.prompt_format.format(
                level=complexity_level,
                question=question
            )
        
        # Create chat messages
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
        
        # Format using the chat template
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Add EOS token to ensure proper generation termination
        formatted_text += tokenizer.eos_token
        
        formatted_data.append({
            "text": formatted_text,
            "context_id": row["context_id"]
        })
    
    # Create dataset
    return datasets.Dataset.from_list(formatted_data)

class ContextAwareSampler(Sampler):
    """Sampler that yields indices ordered by context."""
    
    def __init__(self, dataset):
        """Initialize with dataset containing context IDs."""
        self.dataset = dataset
        self.context_groups = {}
        
        # Group indices by context
        for i, ctx_id in enumerate(dataset["context_id"]):
            if ctx_id not in self.context_groups:
                self.context_groups[ctx_id] = []
            self.context_groups[ctx_id].append(i)
        
        self.context_ids = list(self.context_groups.keys())
        
    def __iter__(self) -> Iterator[int]:
        """Yield indices ordered by context."""
        # Shuffle contexts for randomization
        random.shuffle(self.context_ids)
        
        # Process each context
        for context_id in self.context_ids:
            indices = self.context_groups[context_id]
            random.shuffle(indices)
            yield from indices
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.dataset)

class CustomDataCollator:
    """Custom data collator."""
    
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm = False
    
    def __call__(self, features):
        # Extract text for tokenization
        texts = [f["text"] for f in features]
        
        # Tokenize with padding and truncation
        batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Create labels (for causal language modeling)
        batch["labels"] = batch["input_ids"].clone()
        
        return batch