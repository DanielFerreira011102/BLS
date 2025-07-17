import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from tqdm import tqdm

# Import metric modules
from rq1.metrics.impl.umls import QuickUmlsClassifier
from rq1.metrics.impl.syntax import SyntaxClassifier
from rq1.metrics.impl.commonlit import CommonLitClassifier
from rq1.metrics.impl.masked_prob import MaskedProbabilityClassifier
from rq1.metrics.impl.gispy import GisPyClassifier
from rq1.metrics.impl.scigispy import SciGisPyClassifier
from rq1.metrics.impl.textstat import TextClassifier
from rq1.metrics.impl.cluster import ClusterBasedReadabilityClassifier
from rq1.metrics.impl.jargon import MedicalJargonClassifier
# from rq1.metrics.impl.llm import LLMReadabilityClassifier
from rq1.llmify.llm import LLMReadabilityClassifier

# Utility functions
from utils.helpers import load_json, save_json, setup_logging

# Set up logging
logger = setup_logging()

# Constants for metric types
BATCH_METRICS = {'commonlit', 'syntax', 'textstat', 'cluster', 'llm', 'jargon'}
SINGLE_METRICS = {'gispy', 'scigispy', 'masked_prob', 'umls'}

# Define dataset paths
DATASET_PATHS = {
    'bioasq': '/data/home/djbf/storage/bls/rq2/outputs/phase0/elo/bioasq/answer_variants.json',
    'liveqa': '/data/home/djbf/storage/bls/rq2/outputs/phase0/elo/liveqa/answer_variants.json',
    'medicationqa': '/data/home/djbf/storage/bls/rq2/outputs/phase0/elo/medicationqa/answer_variants.json',
    "medquad": '/data/home/djbf/storage/bls/rq2/outputs/phase0/elo/medquad/answer_variants.json',
    "mediqaans": '/data/home/djbf/storage/bls/rq2/outputs/phase0/elo/mediqaans/answer_variants.json'
}

### Utility Functions for Checkpointing
def serialize_key(key: Tuple[int, bool, Optional[int], Optional[int]]) -> str:
    """Convert a metadata tuple to a string for JSON serialization."""
    entry_idx, is_original, gen_idx, var_idx = key
    gen_str = str(gen_idx) if gen_idx is not None else "-1"
    var_str = str(var_idx) if var_idx is not None else "-1"
    return f"{entry_idx}:{int(is_original)}:{gen_str}:{var_str}"

def deserialize_key(key_str: str) -> Tuple[int, bool, Optional[int], Optional[int]]:
    """Parse a string back into a metadata tuple."""
    parts = key_str.split(':')
    return (
        int(parts[0]),                  # entry_index
        bool(int(parts[1])),            # is_original
        int(parts[2]) if parts[2] != '-1' else None,  # generation_index
        int(parts[3]) if parts[3] != '-1' else None   # variant_index
    )

### Dataset Loading Functions
def load_variants_dataset(path: str) -> List[Dict]:
    """Load a variants dataset from a JSON file."""

    logger.info(f"Loading dataset from {path}")
    data = load_json(path)
    
    if not isinstance(data, list):
        raise ValueError("Dataset must be a list of entries")
    
    logger.info(f"Loaded {len(data)} entries")
    return data

def load_bioasq_variants_dataset() -> List[Dict]:
    """Load the BioASQ dataset."""
    return load_variants_dataset(DATASET_PATHS['bioasq'])

def load_liveqa_variants_dataset() -> List[Dict]:
    """Load the LiveQA dataset."""
    return load_variants_dataset(DATASET_PATHS['liveqa'])

def load_medicationqa_variants_dataset() -> List[Dict]:
    """Load the MedicationQA dataset."""
    return load_variants_dataset(DATASET_PATHS['medicationqa']) 

def load_medquad_variants_dataset() -> List[Dict]:
    """Load the MedQuAD dataset."""
    return load_variants_dataset(DATASET_PATHS['medquad'])

def load_mediqaans_variants_dataset() -> List[Dict]:
    """Load the MediQA-AnS dataset."""
    return load_variants_dataset(DATASET_PATHS['mediqaans'])

### Main Processing Class
class TextReadabilityScorer:
    """Class to calculate readability metrics with checkpointing support."""

    def __init__(self, classifiers: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize with classifiers and optional configuration."""
        self.classifiers = classifiers
        self.enabled_metrics = set(classifiers.keys())
        self.config = config or {}
        logger.info(f"Enabled metrics: {self.enabled_metrics}")

    def calculate_metrics_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Calculate metrics for a batch of texts."""
        results = [{metric: {} for metric in self.enabled_metrics} for _ in texts]

        # Batch-processed metrics
        for metric in BATCH_METRICS & self.enabled_metrics:
            scores = self.classifiers[metric].predict_batch(texts, batch_size=batch_size)
            for i, score in enumerate(scores):
                results[i][metric] = score

        # Single-processed metrics in batches
        for start in tqdm(range(0, len(texts), batch_size), desc="Single metrics"):
            end = min(start + batch_size, len(texts))
            for i in range(start, end):
                for metric in SINGLE_METRICS & self.enabled_metrics:
                    results[i][metric] = self.classifiers[metric].predict_single(texts[i])

        return results

    def process_variants_dataset(
        self,
        json_data: List[Dict],
        dataset_name: str,
        batch_size: int = 32,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1000
    ) -> Dict[str, Any]:
        """Process the dataset with periodic checkpointing."""
        logger.info(f"Processing dataset: {dataset_name}")

        # Load checkpoint if provided and exists
        metrics_dict = self._load_checkpoint(checkpoint_path)

        # Extract all texts and metadata
        all_texts, text_metadata = self.extract_texts(json_data)
        logger.info(f"Total texts: {len(all_texts)}, Processed: {len(metrics_dict)}")

        # Filter unprocessed texts
        remaining_metadata = [meta for meta in text_metadata if meta not in metrics_dict]
        remaining_texts = [all_texts[text_metadata.index(meta)] for meta in remaining_metadata]
        logger.info(f"Remaining texts to process: {len(remaining_texts)}")

        # Process remaining texts in batches
        processed_count = len(metrics_dict)
        for start in tqdm(range(0, len(remaining_texts), batch_size), desc="Processing batches"):
            end = min(start + batch_size, len(remaining_texts))
            batch_texts = remaining_texts[start:end]
            batch_metadata = remaining_metadata[start:end]
            batch_metrics = self.calculate_metrics_batch(batch_texts, batch_size)

            # Update metrics dictionary
            for meta, metrics in zip(batch_metadata, batch_metrics):
                metrics_dict[meta] = metrics

            # Periodic checkpoint save
            processed_count += len(batch_texts)
            if checkpoint_path and processed_count % checkpoint_interval == 0:
                self._save_checkpoint(metrics_dict, checkpoint_path)
                logger.info(f"Checkpoint saved after processing {processed_count} texts")

        # Final checkpoint save
        if checkpoint_path and remaining_texts:
            self._save_checkpoint(metrics_dict, checkpoint_path)
            logger.info(f"Final checkpoint saved with {len(metrics_dict)} processed texts")

        # Map metrics to dataset structure
        processed_data = self.map_metrics_to_dataset(json_data, metrics_dict)
        return {'dataset': dataset_name, 'config': self.config, 'samples': processed_data}

    def _load_checkpoint(self, checkpoint_path: str) -> Dict[Tuple, Dict]:
        """Load a checkpoint file if it exists."""
        metrics_dict = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint_data = load_json(checkpoint_path)
            metrics_dict = {deserialize_key(k): v for k, v in checkpoint_data.items()}
            logger.info(f"Loaded checkpoint with {len(metrics_dict)} processed texts")
        return metrics_dict

    def _save_checkpoint(self, metrics_dict: Dict[Tuple, Dict], checkpoint_path: str) -> None:
        """Save the metrics dictionary to a checkpoint file."""
        checkpoint_data = {serialize_key(k): v for k, v in metrics_dict.items()}
        
        # First write to a temporary file to avoid corruption if interrupted
        temp_path = f"{checkpoint_path}.tmp"
        save_json(checkpoint_data, temp_path)
            
        # Then replace the original file
        Path(temp_path).replace(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def extract_texts(self, json_data: List[Dict]) -> Tuple[List[str], List[Tuple[int, bool, Optional[int], Optional[int]]]]:
        """Extract valid texts and their metadata from the dataset."""
        texts = []
        metadata = [] # Stores (entry_index, is_original, generation_index, variant_index)

        for entry_idx, entry in enumerate(json_data):
            original = entry.get('original_answer', '')
            generations = entry.get('generations', [])

            # Skip invalid entries
            if not original or original == "[CONTENT_MISMATCH]":
                logger.debug(f"Skipping entry {entry.get('question_id', 'unknown')} due to invalid original answer.")
                continue
            
            # Skip entries with no generations
            if not generations:
                logger.debug(f"Skipping entry {entry.get('question_id', 'unknown')} with no generations.")
                continue

            # Check if ANY variant in ANY generation has content mismatch
            has_content_mismatch = False
            for gen in generations:
                variants = gen.get('variants', [])
                if any(v.get('generated_answer', '') == "[CONTENT_MISMATCH]" for v in variants):
                    has_content_mismatch = True
                    logger.debug(f"Skipping entire entry {entry.get('question_id', 'unknown')} due to variant mismatch.")
                    break
            
            # Skip the entire entry if any mismatch found
            if has_content_mismatch:
                continue
                
            # Add original text
            texts.append(original)
            metadata.append((entry_idx, True, None, None))

            # Add all valid variants from all generations
            for gen_idx, gen in enumerate(generations):
                variants = gen.get('variants', [])
                for var_idx, var in enumerate(variants):
                    text = var.get('generated_answer', '')
                    if text and text != "[CONTENT_MISMATCH]":
                        texts.append(text)
                        metadata.append((entry_idx, False, gen_idx, var_idx))

        logger.info(f"Collected {len(texts)} valid texts")
        return texts, metadata

    def map_metrics_to_dataset(self, json_data: List[Dict], metrics_dict: Dict[Tuple, Dict]) -> List[Dict]:
        """Map metrics back to the dataset structure."""
        processed_data = []
        for entry_idx, entry in enumerate(json_data):
            orig_key = (entry_idx, True, None, None)
            if orig_key not in metrics_dict:
                continue

            processed_entry = {
                'question_id': entry.get('question_id', ''),
                'question': entry.get('question', ''),
                'original_answer': {
                    'text': entry['original_answer'],
                    'metrics': metrics_dict[orig_key]
                },
                'generations': []
            }

            for gen_idx, gen in enumerate(entry.get('generations', [])):
                variants = []
                for var_idx, var in enumerate(gen.get('variants', [])):
                    var_key = (entry_idx, False, gen_idx, var_idx)
                    if var_key in metrics_dict:
                        variants.append({
                            'complexity_level': var.get('complexity_level', ''),
                            'text': var['generated_answer'],
                            'metrics': metrics_dict[var_key]
                        })
                if variants:
                    processed_entry['generations'].append({
                        'model_used': gen.get('model_used', ''),
                        'temperature': gen.get('temperature', ''),
                        'raw_response': gen.get('raw_response', ''),
                        'variants': variants
                    })

            processed_data.append(processed_entry)

        return processed_data

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save processing results to a JSON file."""
        save_json(results, output_path)
        logger.info(f"Results saved to {output_path}")


# Classifier initialization functions
def init_textstat_classifier(args: argparse.Namespace) -> Optional[TextClassifier]:
    """Initializes the TextClassifier if not disabled."""
    if args.disable_textstat:
        return None
    
    logger.info("Initializing TextClassifier...")
    return TextClassifier(
        model_name=args.spacy_model_textstat,
        dale_chall_path=args.dale_chall_path,
        spache_path=args.spache_path
    )


def init_quickumls_classifier(args: argparse.Namespace) -> Optional[QuickUmlsClassifier]:
    """Initializes the QuickUmlsClassifier if not disabled."""
    if args.disable_umls:
        return None
    
    logger.info("Initializing UMLS classifier...")
    return QuickUmlsClassifier(
        path_to_quickumls=args.quickumls_path,
        lay_vocab_path=args.lay_vocab_path,
        expert_vocab_path=args.expert_vocab_path,
        chv_file_path=args.chv_path,
        model_name=args.spacy_model_umls
    )


def init_syntax_classifier(args: argparse.Namespace) -> Optional[SyntaxClassifier]:
    """Initializes the SyntaxClassifier if not disabled."""
    if args.disable_syntax:
        return None
    
    logger.info("Initializing syntax classifier...")
    return SyntaxClassifier(model_name=args.spacy_model_syntax)


def init_commonlit_classifier(args: argparse.Namespace) -> Optional[CommonLitClassifier]:
    if args.disable_commonlit:
        return None
    
    logger.info("Initializing CommonLit classifier...")
    return CommonLitClassifier(
        model_type=args.commonlit_model_type,
        model_path=args.commonlit_model_path
    )


def init_masked_prob_classifier(args: argparse.Namespace) -> Optional[MaskedProbabilityClassifier]:
    """Initializes the MaskedProbabilityClassifier if not disabled."""
    if args.disable_masked_prob:
        return None
    
    logger.info(f"Initializing masked probability classifier with method '{args.masked_prob_method}'...")
    return MaskedProbabilityClassifier(
        model_name=args.masked_prob_model,
        seed=42,
        num_masks=args.num_masks,
        mask_fraction=args.mask_fraction,
        method=args.masked_prob_method
    )


def init_gispy_classifier(args: argparse.Namespace) -> Optional[GisPyClassifier]:
    """Initializes the GisPyClassifier if not disabled."""
    if args.disable_gispy:
        return None
    
    logger.info("Initializing GisPy classifier...")
    enabled_metrics = {
        'coref': not args.disable_gispy_coref and args.use_corenlp,
        'pcref': not args.disable_gispy_pcref,
        'pcdc': not args.disable_gispy_pcdc,
        'causality': not args.disable_gispy_causality,
        'concreteness': not args.disable_gispy_concreteness,
        'wordnet': not args.disable_gispy_wordnet
    }
        
    return GisPyClassifier(
        model_name=args.spacy_model_gispy,
        megahr_path=args.megahr_path,
        mrc_path=args.mrc_path,
        use_corenlp=args.use_corenlp,
        corenlp_memory=args.corenlp_memory,
        sentence_model=args.sentence_model,
        enabled_metrics=enabled_metrics
    )


def init_scigispy_classifier(args: argparse.Namespace) -> Optional[SciGisPyClassifier]:
    """Initializes the SciGisPyClassifier if not disabled."""
    if args.disable_scigispy:
        return None

    logger.info("Initializing SciGisPy classifier...")
    enabled_metrics = {
        'hypernymy': not args.disable_scigispy_hypernymy,
        'verb_overlap': not args.disable_scigispy_verb_overlap,
        'cohesion': not args.disable_scigispy_cohesion,
        'ic': not args.disable_scigispy_ic,
    }
    
    return SciGisPyClassifier(
        model_name=args.spacy_model_scigispy,
        fasttext_path=args.fasttext_path,
        biowordvec_path=args.biowordvec_path,
        stats_dir=args.corpus_stats_dir,
        sentence_model=args.sentence_model,
        window_size=args.scigispy_window_size,
        breakpoint_percentile=args.scigispy_breakpoint_percentile,
        enabled_metrics=enabled_metrics
    )


def init_cluster_classifier(args: argparse.Namespace) -> Optional[ClusterBasedReadabilityClassifier]:
    """Initializes the ClusterBasedReadabilityClassifier if not disabled."""
    if args.disable_cluster:
        return None

    logger.info("Initializing cluster-based readability classifier...")
    classifier = ClusterBasedReadabilityClassifier(
        word_embedding_path=args.cluster_biowordvec_path,
        model_name=args.spacy_model_cluster
    )
    
    classifier.load_model(args.cluster_model_path)
    return classifier


def init_llm_classifier(args: argparse.Namespace) -> Optional[LLMReadabilityClassifier]:
    """Initializes the LLMReadabilityClassifier if not disabled."""
    if args.disable_llm:
        return None

    logger.info("Initializing LLM readability classifier...")
    return LLMReadabilityClassifier(model_path=args.llm_model_path)


def init_jargon_classifier(args: argparse.Namespace) -> Optional[MedicalJargonClassifier]:
    """Initializes the MedicalJargonClassifier if not disabled."""
    if args.disable_jargon:
        return None
    
    logger.info("Initializing medical jargon classifier...")
    return MedicalJargonClassifier(
        pretrained_model=args.jargon_pretrained_model,
        model_path=args.jargon_model_path,
        device=args.jargon_device,
        spacy_model=args.spacy_model_jargon
    )


def initialize_classifiers(args: argparse.Namespace) -> Dict[str, Any]:
    """Initialize classifiers based on arguments."""
    classifiers = {
        'textstat': init_textstat_classifier(args),
        'umls': init_quickumls_classifier(args),
        'syntax': init_syntax_classifier(args),
        'commonlit': init_commonlit_classifier(args),
        'masked_prob': init_masked_prob_classifier(args),
        'gispy': init_gispy_classifier(args),
        'scigispy': init_scigispy_classifier(args),
        'cluster': init_cluster_classifier(args),
        'llm': init_llm_classifier(args),
        'jargon': init_jargon_classifier(args)
    }
    
    successful = [m for m, c in classifiers.items() if c]
    failed = [m for m, c in classifiers.items() if not c]
    
    if successful:
        logger.info(f"Initialized classifiers: {successful}")
    
    if failed:
        logger.warning(f"Failed classifiers: {failed}")
    
    return {k: v for k, v in classifiers.items() if v is not None}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with checkpointing options."""
    parser = argparse.ArgumentParser(description='Calculate text readability metrics')

    # ===== GENERAL OPTIONS =====
    general = parser.add_argument_group('General Options')
    general.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    general.add_argument('--output', type=str, default='readability_metrics.json', help='Output file path')
    general.add_argument('--checkpoint-path', type=str, help='Path to checkpoint file')
    general.add_argument('--checkpoint-interval', type=int, default=1000, help='Number of texts between checkpoints')

    # ===== DATASET OPTIONS =====
    dataset = parser.add_argument_group('Dataset Options')
    dataset.add_argument('--dataset', type=str, 
            choices=['bioasq', 'liveqa', 'medicationqa', 'medquad', 'mediqaans', 'custom'], 
            default='bioasq', help='Dataset to process')
    dataset.add_argument('--custom-dataset-path', type=str, help='Path to custom dataset')

    # ===== CLASSIFIER TOGGLES =====
    toggle_group = parser.add_argument_group('Classifier Toggles')
    toggle_group.add_argument('--disable-textstat', action='store_true',
                              help='Disable textstat metrics')
    toggle_group.add_argument('--disable-umls', action='store_true',
                              help='Disable UMLS metrics')
    toggle_group.add_argument('--disable-syntax', action='store_true',
                              help='Disable syntax metrics')
    toggle_group.add_argument('--disable-commonlit', action='store_true',
                              help='Disable CommonLit metrics')
    toggle_group.add_argument('--disable-masked-prob', action='store_true',
                              help='Disable masked probability metrics')
    toggle_group.add_argument('--disable-gispy', action='store_true',
                              help='Disable GisPy metrics')
    toggle_group.add_argument('--disable-scigispy', action='store_true',
                              help='Disable SciGisPy metrics')
    toggle_group.add_argument('--disable-cluster', action='store_true',
                              help='Disable cluster-based readability metrics')
    toggle_group.add_argument('--disable-llm', action='store_true',
                              help='Disable LLM readability metrics')
    toggle_group.add_argument('--disable-jargon', action='store_true',
                              help='Disable medical jargon metrics')

    # ===== SPACY MODELS =====
    spacy_group = parser.add_argument_group('spaCy Model Options')
    spacy_group.add_argument('--spacy-model-textstat', type=str, default='en_core_web_trf',
                             help='spaCy model for TextClassifier')
    spacy_group.add_argument('--spacy-model-umls', type=str, default='en_core_web_trf',
                             help='spaCy model for UMLS classifier')
    spacy_group.add_argument('--spacy-model-syntax', type=str, default='en_core_web_trf',
                             help='spaCy model for Syntax classifier')
    spacy_group.add_argument('--spacy-model-gispy', type=str, default='en_core_web_trf',
                             help='spaCy model for GisPy classifier')
    spacy_group.add_argument('--spacy-model-scigispy', type=str, default='en_core_web_trf',
                             help='spaCy model for SciGisPy classifier')
    spacy_group.add_argument('--spacy-model-cluster', type=str, default='en_core_web_trf',
                             help='spaCy model for cluster-based readability classifier')
    spacy_group.add_argument('--spacy-model-jargon', type=str, default='en_core_web_trf',
                             help='spaCy model for medical jargon classifier')

    # ===== TEXTSTAT OPTIONS =====
    textstat_group = parser.add_argument_group('TextStat Options')
    textstat_group.add_argument('--dale-chall-path', type=str,
                               default='/data/home/djbf/storage/bls/resources/datasets/wordlists/dale_chall.txt',
                               help='Path to Dale-Chall word list')
    textstat_group.add_argument('--spache-path', type=str,
                               default='/data/home/djbf/storage/bls/resources/datasets/wordlists/spache.txt',
                               help='Path to Spache word list')

    # ===== UMLS OPTIONS =====
    umls_group = parser.add_argument_group('UMLS Options')
    umls_group.add_argument('--quickumls-path', type=str,
                           default='/data/home/djbf/storage/bls/resources/QuickUMLS',
                           help='Path to QuickUMLS directory')
    umls_group.add_argument('--lay-vocab-path', type=str,
                           default='/data/home/djbf/storage/bls/resources/vocabularies/medlineplus+chv_cuis.pkl',
                           help='Path to lay vocabulary pickle file')
    umls_group.add_argument('--expert-vocab-path', type=str,
                           default='/data/home/djbf/storage/bls/resources/vocabularies/snomed_cuis.pkl',
                           help='Path to expert vocabulary pickle file')
    umls_group.add_argument('--chv-path', type=str,
                           default='/data/home/djbf/storage/bls/resources/datasets/chv/chv_comparison_20250323_033347_xgboost/crawl_xgboost_chv_imputed.csv', 
                           help='Path to OAC CHV dataset file')

    # ===== MASKED PROBABILITY OPTIONS =====
    masked_prob_group = parser.add_argument_group('Masked Probability Options')

    masked_prob_group = parser.add_argument_group('Masked Probability Options')
    masked_prob_group.add_argument('--masked-prob-model', type=str, 
                                  default='emilyalsentzer/Bio_ClinicalBERT', 
                                  help='Model name for masked probability metric')
    masked_prob_group.add_argument('--masked-prob-method', type=str,
                                  choices=['random', 'np', 'rnptc'],
                                  default='rnptc',
                                  help='Method for masked probability: random (default), np (noun phrase), or rnptc (ranked NP, best per paper)')
    masked_prob_group.add_argument('--num-masks', type=int, default=10, 
                                  help='Number of masks to use')
    masked_prob_group.add_argument('--mask-fraction', type=float, default=0.15, 
                                  help='Fraction of text to mask')

    # ===== COMMONLIT OPTIONS =====
    commonlit_group = parser.add_argument_group('CommonLit Options')
    commonlit_group.add_argument('--commonlit-model-path', type=str,
                                default='/data/home/djbf/storage/bls/resources/models/albert-xxlarge-all-data',
                                help='Path to CommonLit model directory')
    commonlit_group.add_argument('--commonlit-model-type', type=str,
                                choices=['deberta_ensemble', 'deberta_single', 'albert'],
                                default='albert',
                                help='Type of CommonLit model to use')

    # ===== GISPY OPTIONS =====
    gispy_group = parser.add_argument_group('GisPy Options')
    gispy_group.add_argument('--disable-gispy-coref', action='store_true',
                            help='Disable coreference resolution')
    gispy_group.add_argument('--disable-gispy-pcref', action='store_true',
                            help='Disable referential cohesion')
    gispy_group.add_argument('--disable-gispy-pcdc', action='store_true',
                            help='Disable deep cohesion')
    gispy_group.add_argument('--disable-gispy-causality', action='store_true',
                            help='Disable causality metrics')
    gispy_group.add_argument('--disable-gispy-concreteness', action='store_true',
                            help='Disable concreteness metrics')
    gispy_group.add_argument('--disable-gispy-wordnet', action='store_true',
                            help='Disable WordNet-based metrics')
    gispy_group.add_argument('--megahr-path', type=str,
                            default='/data/home/djbf/storage/bls/resources/datasets/megahr/megahr.en.sort.i.txt',
                            help='Path to MegaHR dictionary file')
    gispy_group.add_argument('--mrc-path', type=str,
                            default='/data/home/djbf/storage/bls/resources/datasets/mrc/mrc_psycholinguistic_database.csv',
                            help='Path to MRC database file')

    # ===== SCIGISPY OPTIONS =====
    scigispy_group = parser.add_argument_group('SciGisPy Options')
    scigispy_group.add_argument('--disable-scigispy-hypernymy', action='store_true',
                               help='Disable hypernymy metrics')
    scigispy_group.add_argument('--disable-scigispy-verb-overlap', action='store_true',
                               help='Disable verb overlap metrics')
    scigispy_group.add_argument('--disable-scigispy-cohesion', action='store_true',
                               help='Disable cohesion metrics')
    scigispy_group.add_argument('--disable-scigispy-ic', action='store_true',
                               help='Disable information content metrics')
    scigispy_group.add_argument('--fasttext-path', type=str,
                               default='/data/home/djbf/storage/bls/resources/models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin',
                               help='Path to FastText embedding model')
    scigispy_group.add_argument('--biowordvec-path', type=str,
                               default='/data/home/djbf/storage/bls/resources/models/BioWordVec/BioWordVec_PubMed_MIMICIII_d200.bin',
                               help='Path to BioWordVec embedding model')
    scigispy_group.add_argument('--corpus-stats-dir', type=str,
                               default='/data/home/djbf/storage/bls/resources/datasets/ic/corpus_stats_without_pubmed', 
                               help='Path to corpus statistics directory')
    scigispy_group.add_argument('--scigispy-window-size', type=int, default=1,
                               help='Window size for semantic chunking')
    scigispy_group.add_argument('--scigispy-breakpoint-percentile', type=int, default=75,
                               help='Percentile threshold for chunk breakpoints')

    # ===== CLUSTER READABILITY OPTIONS =====
    cluster_group = parser.add_argument_group('Cluster-based Readability Options')
    cluster_group.add_argument('--cluster-model-path', type=str,
                              default='/data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/clear',
                              help='Path to cluster-based readability model directory')
    cluster_group.add_argument('--cluster-biowordvec-path', type=str,
                              default='/data/home/djbf/storage/bls/resources/models/BioWordVec/BioWordVec_PubMed_MIMICIII_d200.bin',
                              help='Path to BioWordVec embedding model')

    # ===== LLM OPTIONS =====
    llm_group = parser.add_argument_group('LLM Options')
    llm_group.add_argument('--llm-model-path', type=str,
                          default='/data/home/djbf/storage/bls/rq2/outputs/llmify/default/improved/mse_1.0+kl_1.0+ce_0.0+soft_0.0',
                          help='Path to LLM model for readability estimation')

    # ===== MEDICAL JARGON OPTIONS =====
    jargon_group = parser.add_argument_group('Medical Jargon Options')
    jargon_group.add_argument('--jargon-pretrained-model', type=str, 
                            default='roberta-large',
                            help='Pretrained model for medical jargon classifier')
    jargon_group.add_argument('--jargon-model-path', type=str,
                            default='/data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model',
                            help='Path to medical jargon model directory')
    jargon_group.add_argument('--jargon-device', type=str, default=None,
                            help='Device to run the jargon model on (optional)')

    # ===== SHARED RESOURCES =====
    shared_group = parser.add_argument_group('Shared Resources')
    shared_group.add_argument('--sentence-model', type=str,
                             default='kamalkraj/BioSimCSE-BioLinkBERT-BASE',
                             help='Model name for sentence embeddings')

    # ===== CORENLP CONFIGURATION =====
    corenlp_group = parser.add_argument_group('CoreNLP Configuration')
    corenlp_group.add_argument('--use-corenlp', action='store_true', default=False,
                              help='Enable CoreNLP for coreference resolution')
    corenlp_group.add_argument('--corenlp-memory', type=str, default='4G',
                              help='Memory allocation for CoreNLP server')

    return parser.parse_args()


def main() -> None:
    """Execute the processing pipeline with checkpointing."""
    args = parse_arguments()

    # Convert args to dictionary to store in results
    config_dict = vars(args)

    # Initialize classifiers
    classifiers = initialize_classifiers(args)

    # Filter enabled classifiers
    enabled_classifiers = {metric: classifier for metric, classifier in classifiers.items() if classifier is not None}
    
    # Initialize scorer with config
    scorer = TextReadabilityScorer(enabled_classifiers, config=config_dict)

    # Load dataset
    if args.dataset == 'bioasq':
        json_data = load_bioasq_variants_dataset()
    elif args.dataset == 'liveqa':
        json_data = load_liveqa_variants_dataset()
    elif args.dataset == 'medicationqa':
        json_data = load_medicationqa_variants_dataset()
    elif args.dataset == 'medquad':
        json_data = load_medquad_variants_dataset()
    elif args.dataset == 'mediqaans':
        json_data = load_mediqaans_variants_dataset()
    else:
        if not args.custom_dataset_path:
            raise ValueError("Custom dataset path must be provided with --custom-dataset-path")
        json_data = load_variants_dataset(args.custom_dataset_path)

    # TESTING ONLY: Limit to first some samples
    # MAX_SAMPLES = 200  # Change this number as needed
    # json_data = json_data[:MAX_SAMPLES]
    # logger.info(f"TESTING MODE: Limited to {MAX_SAMPLES} samples")

    # Process with checkpointing
    results = {
        args.dataset: scorer.process_variants_dataset(
            json_data,
            args.dataset,
            args.batch_size,
            args.checkpoint_path,
            args.checkpoint_interval
        )
    }

    # Save results
    scorer.save_results(results, args.output)
    logger.info(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()