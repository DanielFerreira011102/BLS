import numpy as np
import spacy
import textstat
from typing import List, Tuple
from dataclasses import dataclass, field
import pandas as pd
from collections import Counter
import multiprocessing
from tqdm import tqdm
import argparse
import json
import sys
import csv
from pathlib import Path

@dataclass
class TextComplexityConfig:
    """Configuration for text-level complexity metrics"""
    fk_weight: float = 0.5
    smog_weight: float = 0.5

@dataclass
class SyntaxComplexityConfig:
    """Configuration for syntax-level complexity metrics"""
    content_ratio_weight: float = 0.5
    noun_ratio_weight: float = 0.5

@dataclass
class CHELCConfig:
    """Overall CHELC configuration"""
    text_weight: float = 0.25
    syntax_weight: float = 0.25
    term_weight: float = 0.25
    semantic_weight: float = 0.25
    normalize: bool = True
    n_jobs: int = multiprocessing.cpu_count()
    batch_size: int = 1000
    text_config: TextComplexityConfig = field(default_factory=TextComplexityConfig)
    syntax_config: SyntaxComplexityConfig = field(default_factory=SyntaxComplexityConfig)

@dataclass
class ComplexityScores:
    """Container for complexity scores with detailed metrics"""
    text: str
    text_metrics: np.ndarray
    syntax_metrics: np.ndarray
    term_level: float
    semantic_level: float
    overall: float
    config: CHELCConfig
    
    def to_dict(self) -> dict:
        """Convert scores to dictionary for JSON serialization with complete information"""
        return {
            'text': self.text,
            'metrics': {
                'text_metrics': {
                    'fk_score': float(self.text_metrics[0]),
                    'smog_score': float(self.text_metrics[1]),
                    'text_level': float(self.text_metrics[2])
                },
                'syntax_metrics': {
                    'content_ratio': float(self.syntax_metrics[0]),
                    'noun_ratio': float(self.syntax_metrics[1]),
                    'syntax_level': float(self.syntax_metrics[2])
                },
                'term_level': float(self.term_level),
                'semantic_level': float(self.semantic_level),
                'overall': float(self.overall)
            },
            'configuration': {
                'weights': {
                    'text_weight': float(self.config.text_weight),
                    'syntax_weight': float(self.config.syntax_weight),
                    'term_weight': float(self.config.term_weight),
                    'semantic_weight': float(self.config.semantic_weight)
                },
                'normalize': self.config.normalize,
                'text_config': {
                    'fk_weight': float(self.config.text_config.fk_weight),
                    'smog_weight': float(self.config.text_config.smog_weight),
                },
                'syntax_config': {
                    'content_ratio_weight': float(self.config.syntax_config.content_ratio_weight),
                    'noun_ratio_weight': float(self.config.syntax_config.noun_ratio_weight)
                },
                'processing': {
                    'n_jobs': self.config.n_jobs,
                    'batch_size': self.config.batch_size
                }
            }
        }

class TextInputHandler:
    """Handles different text input formats"""
    
    @staticmethod
    def read_csv(file_path: str, text_column: str) -> List[str]:
        """Read texts from CSV file"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
            if text_column not in df.columns:
                raise ValueError(f"Column '{text_column}' not found in CSV file")
            return df[text_column].fillna('').astype(str).tolist()
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {str(e)}")

    @staticmethod
    def read_text_file(file_path: str) -> List[str]:
        """Read texts from text file, handling quoted and escaped text properly"""
        texts = []
        current_text = []
        in_quotes = False
        prev_char = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_line = line  # Keep the original line with newline
                line = line.rstrip()  # Only remove trailing whitespace
                
                # Process the line character by character
                i = 0
                while i < len(line):
                    char = line[i]
                    
                    if char == '\\' and i + 1 < len(line) and line[i + 1] == '"':
                        # Found escaped quote, preserve it
                        if current_text:
                            current_text.append('"')
                        else:
                            current_text = ['"']
                        i += 2  # Skip both backslash and quote
                        continue
                        
                    if char == '"' and prev_char != '\\':
                        # Found unescaped quote
                        in_quotes = not in_quotes
                        if not in_quotes:  # End of quoted text
                            if current_text:
                                texts.append(''.join(current_text))
                            current_text = []
                    else:
                        # Regular character
                        if in_quotes:
                            current_text.append(char)
                        else:
                            if char != '"':  # Ignore quotes outside quote blocks
                                if not current_text:
                                    current_text = [char]
                                else:
                                    current_text.append(char)
                    
                    prev_char = char
                    i += 1
                
                # Handle line endings
                if in_quotes:
                    if current_text:
                        current_text.append('\n')  # Preserve newline in quoted text
                else:
                    if current_text:
                        text = ''.join(current_text)
                        if text:
                            texts.append(text)
                        current_text = []
        
        # Handle any remaining text
        if current_text:
            text = ''.join(current_text)
            if text:
                texts.append(text)
        
        return texts

    @staticmethod
    def process_text_string(text: str) -> List[str]:
        """Process a single text string"""
        return [text] if text else []

class ParallelCHELCMetrics:
    """Parallelized CHELC implementation"""
    
    def __init__(self, config: CHELCConfig):
        self.config = config
        self.nlp = self._initialize_spacy()
        self.content_pos = {'NOUN', 'ADJ', 'VERB', 'ADV'}
        self.noun_pos = {'NOUN'}

    def _initialize_spacy(self):
        """Initialize spaCy with optimized settings"""
        nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'lemmatizer'])
        nlp.enable_pipe('tagger')
        return nlp

    @staticmethod
    def _process_text_chunk(texts: List[str]) -> List[Tuple[float, float]]:
        """Process a chunk of texts for readability metrics"""
        return [(textstat.flesch_kincaid_grade(text), 
                textstat.smog_index(text)) for text in texts]

    @staticmethod
    def _process_syntax_chunk(nlp, texts: List[str]) -> List[Tuple[float, float]]:
        """Process a chunk of texts for syntax metrics"""
        content_pos = {'NOUN', 'ADJ', 'VERB', 'ADV'}
        noun_pos = {'NOUN'}
        metrics = []
        docs = nlp.pipe(texts)
        
        for doc in docs:
            pos_counts = Counter(token.pos_ for token in doc 
                               if not token.is_punct and not token.is_space)
            total_words = sum(pos_counts.values())
            if total_words == 0:
                metrics.append((0, 0))
                continue
            
            content_words = sum(pos_counts[pos] for pos in content_pos)
            noun_count = sum(pos_counts[pos] for pos in noun_pos)
            content_ratio = content_words / total_words
            noun_ratio = noun_count / total_words
            metrics.append((content_ratio, noun_ratio))
        
        return metrics

    def calculate_text_complexity(self, texts: List[str]) -> np.ndarray:
        """Parallel text complexity calculation"""
        # Process in smaller batches
        batch_size = min(100, self.config.batch_size)
        chunks = [texts[i:i + batch_size] 
                for i in range(0, len(texts), batch_size)]
        
        all_results = []
        for chunk in tqdm(chunks, desc="Processing text complexity"):
            result = self._process_text_chunk(chunk)
            all_results.extend(result)
        
        if not all_results:
            return np.zeros((len(texts), 3))
        
        fk_scores, smog_scores = zip(*all_results)
        fk_scores = np.array(fk_scores, dtype=np.float64)
        smog_scores = np.array(smog_scores, dtype=np.float64)
        
        # Use global normalization setting
        fk_norm = self._normalize_ranks(fk_scores)
        smog_norm = self._normalize_ranks(smog_scores)
        
        weights = np.array([self.config.text_config.fk_weight, 
                        self.config.text_config.smog_weight])
        text_level = np.average([fk_norm, smog_norm], weights=weights, axis=0)
        
        return np.column_stack((fk_norm, smog_norm, text_level))

    def calculate_syntax_complexity(self, texts: List[str]) -> np.ndarray:
        """Parallel syntax complexity calculation"""
        batch_size = min(100, self.config.batch_size)
        chunks = [texts[i:i + batch_size] 
                for i in range(0, len(texts), batch_size)]
        
        all_results = []
        for chunk in tqdm(chunks, desc="Processing syntax complexity"):
            result = self._process_syntax_chunk(self.nlp, chunk)
            all_results.extend(result)
        
        if not all_results:
            return np.zeros((len(texts), 3))
        
        content_ratios, noun_ratios = zip(*all_results)
        content_ratios = np.array(content_ratios, dtype=np.float64)
        noun_ratios = np.array(noun_ratios, dtype=np.float64)
        
        # Use global normalization setting
        content_norm = self._normalize_ranks(content_ratios)
        noun_norm = self._normalize_ranks(noun_ratios)
        
        weights = np.array([self.config.syntax_config.content_ratio_weight,
                        self.config.syntax_config.noun_ratio_weight])
        syntax_level = np.average([content_norm, noun_norm], weights=weights, axis=0)
        
        return np.column_stack((content_norm, noun_norm, syntax_level))

    def _normalize_ranks(self, values: np.ndarray) -> np.ndarray:
        """Vectorized rank normalization"""
        if not self.config.normalize:
            return values
        ranks = pd.Series(values).rank(method='average')
        return np.array(ranks.values / len(values), dtype=np.float64)

    def calculate_chelc_scores(self, texts: List[str]) -> List[ComplexityScores]:
        """Calculate all CHELC scores using parallel processing"""
        text_metrics = self.calculate_text_complexity(texts)
        syntax_metrics = self.calculate_syntax_complexity(texts)
        
        term_level = np.zeros(len(texts))
        semantic_level = np.zeros(len(texts))
        
        weights = np.array([self.config.text_weight, self.config.syntax_weight,
                          self.config.term_weight, self.config.semantic_weight])
        
        component_scores = np.column_stack([
            text_metrics[:, 2],
            syntax_metrics[:, 2],
            term_level,
            semantic_level
        ])
        
        overall_scores = np.average(component_scores, weights=weights, axis=1)
        
        return [
            ComplexityScores(
                text=texts[i],
                text_metrics=text_metrics[i],
                syntax_metrics=syntax_metrics[i],
                term_level=term_level[i],
                semantic_level=semantic_level[i],
                overall=overall_scores[i],
                config=self.config
            )
            for i in range(len(texts))
        ]

def load_texts(input_source: str, input_type: str = 'text', text_column: str = None) -> List[str]:
    """
    Load texts from various input sources
    
    Args:
        input_source: Path to input file or text string
        input_type: Type of input ('text', 'csv', or 'string')
        text_column: Column name for CSV input
    
    Returns:
        List of texts
    """
    handler = TextInputHandler()
    
    if input_type == 'csv':
        if not text_column:
            raise ValueError("text_column must be specified for CSV input")
        return handler.read_csv(input_source, text_column)
    elif input_type == 'text':
        return handler.read_text_file(input_source)
    elif input_type == 'string':
        return handler.process_text_string(input_source)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")

def save_results(scores: List[ComplexityScores], output_file: str, overwrite: bool = False):
    """Save results to file (JSON or CSV) with append support"""
    results = [score.to_dict() for score in scores]
    
    if output_file.lower().endswith('.csv'):
        # Handle CSV output
        flat_results = []
        for result in results:
            flat_dict = {
                'text': result['text'].replace('\n', '\\n').replace('"', '\\"')
            }
            # Flatten metrics
            for metric_key, metric_value in result['metrics'].items():
                if isinstance(metric_value, dict):
                    for sub_key, sub_value in metric_value.items():
                        flat_dict[f"{metric_key}_{sub_key}"] = sub_value
                else:
                    flat_dict[metric_key] = metric_value
            
            # Flatten configuration
            for config_key, config_value in result['configuration'].items():
                if isinstance(config_value, dict):
                    for sub_key, sub_value in config_value.items():
                        flat_dict[f"config_{config_key}_{sub_key}"] = sub_value
                else:
                    flat_dict[f"config_{config_key}"] = config_value
            
            flat_results.append(flat_dict)
        
        df = pd.DataFrame(flat_results)
        
        if Path(output_file).exists() and not overwrite:
            # Append to existing CSV
            df.to_csv(output_file, mode='a', header=not Path(output_file).exists(), index=False)
        else:
            # Create new CSV
            df.to_csv(output_file, index=False)
    else:
        # Handle JSON output
        if Path(output_file).exists() and not overwrite:
            try:
                # Read existing JSON
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
                if not isinstance(existing_results, list):
                    existing_results = [existing_results]
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
            
            # Append new results
            existing_results.extend(results)
            results = existing_results
        
        # Write JSON (either new or combined)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Calculate CHELC scores for texts')
    
    # Input/Output options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-file', help='Input file path (CSV or text)')
    input_group.add_argument('--input-text', help='Direct text input')
    
    parser.add_argument('--output', '-o', help='Output file (JSON or CSV)')
    parser.add_argument('--input-type', choices=['csv', 'text', 'string'],
                       help='Input type (if not specified, will be inferred)')
    parser.add_argument('--text-column', help='Column name for CSV input')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite output file if it exists (default: append)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable global score normalization for all metrics')
    
    # Processing options
    parser.add_argument('--jobs', '-j', type=int, default=multiprocessing.cpu_count(),
                       help='Number of parallel jobs')
    parser.add_argument('--batch-size', '-b', type=int, default=1000,
                       help='Batch size for processing')
    
    # Weight configurations
    parser.add_argument('--text-weight', type=float, default=0.25,
                       help='Weight for text-level complexity')
    parser.add_argument('--syntax-weight', type=float, default=0.25,
                       help='Weight for syntax-level complexity')
    parser.add_argument('--term-weight', type=float, default=0.25,
                       help='Weight for term-level complexity')
    parser.add_argument('--semantic-weight', type=float, default=0.25,
                       help='Weight for semantic-level complexity')
    
    # Metric-specific options
    parser.add_argument('--fk-weight', type=float, default=0.5,
                       help='Weight for Flesch-Kincaid score')
    parser.add_argument('--smog-weight', type=float, default=0.5,
                       help='Weight for SMOG index')
    
    args = parser.parse_args()
    
    # Create configurations with global normalization
    text_config = TextComplexityConfig(
        fk_weight=args.fk_weight,
        smog_weight=args.smog_weight
    )
    
    syntax_config = SyntaxComplexityConfig()
    
    config = CHELCConfig(
        text_weight=args.text_weight,
        syntax_weight=args.syntax_weight,
        term_weight=args.term_weight,
        semantic_weight=args.semantic_weight,
        normalize=not args.no_normalize,  # Global normalization setting
        n_jobs=1 if args.input_type == 'string' else args.jobs,
        batch_size=args.batch_size,
        text_config=text_config,
        syntax_config=syntax_config
    )
    
    try:
        input_source = args.input_text if args.input_text else args.input_file
        texts = load_texts(input_source, args.input_type, args.text_column)
        print(f"Loaded {len(texts)} texts")
        
        chelc = ParallelCHELCMetrics(config)
        scores = chelc.calculate_chelc_scores(texts)
        
        if args.output:
            save_results(scores, args.output, overwrite=args.overwrite)
            print(f"Results {'saved to' if args.overwrite else 'appended to'} {args.output}")
        else:
            # Print results to stdout
            for i, score in enumerate(scores, 1):
                print(f"\nText {i} scores:")
                print(json.dumps(score.to_dict(), indent=2))
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()