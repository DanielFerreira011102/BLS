import json
import math
from collections import Counter
from pathlib import Path
from typing import List

import spacy

from utils.helpers import setup_logging

logger = setup_logging()

class CorpusStats:
    """Container for corpus statistics with methods for IC calculations."""
    
    def __init__(self, term_counts: Counter, total_terms: int, unique_terms: int):
        self.term_counts = term_counts
        self.total_terms = total_terms
        self.vocabulary_size = unique_terms  # Use provided value instead of calculating
    
    def get_term_count(self, term: str) -> int:
        """Get the count of a term in the corpus."""
        return self.term_counts.get(term.lower(), 0)
    
    def get_term_probability(self, term: str) -> float:
        """Calculate probability of a term with smoothing."""
        count = self.get_term_count(term) + 1  # Add-1 smoothing
        return count / (self.total_terms + self.vocabulary_size)
    
    def get_term_ic(self, term: str) -> float:
        """Calculate Information Content for a term."""
        prob = self.get_term_probability(term)
        return -math.log(prob)
    
    def merge_with(self, other: 'CorpusStats') -> 'CorpusStats':
        """Merge this CorpusStats with another one."""
        # Merge term counts
        merged_counts = Counter(self.term_counts)
        merged_counts.update(other.term_counts)
        
        # Create new merged stats
        return CorpusStats(
            term_counts=merged_counts,
            total_terms=self.total_terms + other.total_terms,
            unique_terms=len(merged_counts)
        )

    @classmethod
    def load(cls, stats_dir: str) -> 'CorpusStats':
        """Load corpus statistics from directory."""        
        stats_path = Path(stats_dir)
        
        # Load metadata first to get unique_terms
        meta_file = stats_path / "metadata.json"
        with open(meta_file) as f:
            metadata = json.load(f)
            total_terms = metadata['total_terms']
            unique_terms = metadata['unique_terms']
            
        # Load term counts
        counts_file = stats_path / "term_counts.json"
        with open(counts_file) as f:
            counts_dict = json.load(f)
            term_counts = Counter(counts_dict)
            
        return cls(term_counts, total_terms, unique_terms)
    
    def save(self, output_dir: str) -> None:
        """Save corpus statistics to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save term counts
        counts_file = output_path / "term_counts.json"
        with open(counts_file, 'w') as f:
            json.dump(dict(self.term_counts), f)
            
        # Save metadata
        meta_file = output_path / "metadata.json"
        with open(meta_file, 'w') as f:
            json.dump({
                'total_terms': self.total_terms,
                'unique_terms': self.vocabulary_size,
                'top_terms': dict(self.term_counts.most_common(100))
            }, f, indent=2)

class CorpusCalculator:
    """Calculates Information Content statistics for a corpus."""
    
    def __init__(self, model_name: str = 'en_core_web_trf'):
        logger.info(f"Loading spaCy model {model_name}...")
        spacy.prefer_gpu()  # Use GPU if available

        self.nlp = spacy.load(model_name)
        
        # Initialize counters
        self.term_counts = Counter()
        self.total_terms = 0
        self.unique_terms = 0
        
    def process_texts(self, texts: List[str], batch_size: int = 1000) -> None:
        """Process texts to build term frequency statistics."""
        logger.info(f"Processing {len(texts)} texts...")
        
        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(texts) - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process batch with spaCy
            for doc in self.nlp.pipe(texts=batch, disable=["parser", "ner"], batch_size=batch_size):
                # Count nouns and verbs
                for token in doc:
                    if token.pos_ in ['NOUN', 'VERB']:
                        lemma = token.lemma_.lower()
                        self.term_counts[lemma] += 1
                        self.total_terms += 1
        
        self.unique_terms = len(self.term_counts)
        logger.info(f"Processed {self.total_terms} terms")
        logger.info(f"Found {self.unique_terms} unique terms")
    
    def save_stats(self, output_dir: str) -> None:
        """Save corpus statistics using CorpusStats."""
        stats = CorpusStats(
            term_counts=self.term_counts,
            total_terms=self.total_terms,
            unique_terms=self.unique_terms
        )
        stats.save(output_dir)
        logger.info(f"Saved corpus statistics to {output_dir}")

def main():
    """Main function to build corpus and calculate statistics."""
    import argparse
    parser = argparse.ArgumentParser(description='Calculate corpus statistics')
    parser.add_argument('--output-dir', type=str, default='corpus_stats',
                      help='Directory to save statistics')
    parser.add_argument('--spacy-model', type=str, default='en_core_web_trf',
                      help='SpaCy model to use')
    parser.add_argument('--min-words', type=int, default=0,
                      help='Minimum words per text')
    parser.add_argument('--batch-size', type=int, default=1000,
                      help='Batch size for processing')
    args = parser.parse_args()

    # Build corpus
    from ic_corpus_builder import CorpusBuilder
    builder = CorpusBuilder(min_words=args.min_words)
    logger.info("Building corpus...")
    corpus = builder.build_corpus()
    
    # Calculate statistics
    calculator = CorpusCalculator(model_name=args.spacy_model)
    calculator.process_texts(corpus, batch_size=args.batch_size)
    
    # Save results
    calculator.save_stats(args.output_dir)

if __name__ == "__main__":
    main()