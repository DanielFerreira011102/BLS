import os
import random
import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def setup_logging(level=logging.INFO, format_str=None):
    """Set up logging configuration."""
    if format_str is None:
        format_str = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        
    logging.basicConfig(
        level=level,
        format=format_str
    )
    return logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.ndarray, list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    if isinstance(obj, dict):
        # Convert both keys and values
        return {
            convert_numpy_types(key): convert_numpy_types(value)
            for key, value in obj.items()
        }
    if isinstance(obj, set):
        return list(obj)
    
    return obj


def save_json(data: Any, filepath: Union[str, Path], indent: int = 2):
    """Save data to a JSON file with proper type conversion."""
    # Convert to native Python types for JSON serialization
    converted_data = convert_numpy_types(data)
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict:
    """Load data from a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.getLogger(__name__).info(f"Random seed set to {seed}")