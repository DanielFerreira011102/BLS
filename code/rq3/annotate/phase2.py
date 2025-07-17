import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

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
    'baseline': '/data/home/djbf/storage/bls/rq3/outputs/phase1/baseline/answer_variants.json',
    'fewshot': '/data/home/djbf/storage/bls/rq3/outputs/phase1/fewshot/answer_variants.json',
    'finetuned': '/data/home/djbf/storage/bls/rq3/outputs/phase1/finetuned/answer_variants.json',
    'finetuned3b': '/data/home/djbf/storage/bls/rq3/outputs/phase1/finetuned3b/answer_variants.json',
    'finetunedIt': '/data/home/djbf/storage/bls/rq3/outputs/phase1/finetunedIt/answer_variants.json',
    'claude': '/data/home/djbf/storage/bls/rq3/outputs/phase1/claude/answer_variants.json',
}

class TextReadabilityScorer:
    """Class to calculate readability metrics."""

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

    def process_file(self, input_path: str, batch_size: int = 32, dataset_name: str = None) -> Dict[str, Any]:
        """Process a file containing responses."""
        logger.info(f"Processing file: {input_path}")
        
        # Determine dataset name if not provided
        if dataset_name is None:
            dataset_name = Path(input_path).stem
            
        # Load data
        data = load_json(input_path)
        if not isinstance(data, list):
            logger.error("Input file must contain a list of entries")
            return {}
            
        logger.info(f"Loaded {len(data)} entries")
        
        # Extract all texts for batch processing
        texts = [entry.get("response", "") for entry in data]
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if not text:
                logger.debug(f"Skipping entry {i} due to empty response")
                continue
            valid_texts.append(text)
            valid_indices.append(i)
            
        logger.info(f"Processing {len(valid_texts)} valid texts")
        
        # Calculate metrics
        metrics_results = self.calculate_metrics_batch(valid_texts, batch_size)
        
        # Add metrics to original data
        for result_idx, data_idx in enumerate(valid_indices):
            data[data_idx]["metrics"] = metrics_results[result_idx]
            
        # Return structured results in the desired format
        return {
            dataset_name: {
                "dataset": dataset_name,
                "config": self.config,
                "samples": data
            }
        }
        
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save processing results to a JSON file."""
        save_json(results, output_path)
        logger.info(f"Results saved to {output_path}")


# Classifier initialization functions
def init_textstat_classifier(args: argparse.Namespace) -> Optional[TextClassifier]:
    """Initialize the TextClassifier if not disabled."""
    if args.disable_textstat:
        return None
        
    logger.info("Initializing TextClassifier...")
    return TextClassifier(
        model_name=args.spacy_model_textstat,
        dale_chall_path=args.dale_chall_path,
        spache_path=args.spache_path
    )

def init_quickumls_classifier(args: argparse.Namespace) -> Optional[QuickUmlsClassifier]:
    """Initialize the QuickUmlsClassifier if not disabled."""
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
    """Initialize the SyntaxClassifier if not disabled."""
    if args.disable_syntax:
        return None
        
    logger.info("Initializing syntax classifier...")
    return SyntaxClassifier(model_name=args.spacy_model_syntax)

def init_commonlit_classifier(args: argparse.Namespace) -> Optional[CommonLitClassifier]:
    """Initialize the CommonLitClassifier if not disabled."""
    if args.disable_commonlit:
        return None
        
    logger.info("Initializing CommonLit classifier...")
    return CommonLitClassifier(
        model_type=args.commonlit_model_type,
        model_path=args.commonlit_model_path
    )

def init_masked_prob_classifier(args: argparse.Namespace) -> Optional[MaskedProbabilityClassifier]:
    """Initialize the MaskedProbabilityClassifier if not disabled."""
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
    """Initialize the GisPyClassifier if not disabled."""
    if args.disable_gispy:
        return None
        
    logger.info("Initializing GisPy classifier...")
    enabled_metrics = {
        'coref': not args.disable_gispy_coref,
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
    """Initialize the SciGisPyClassifier if not disabled."""
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
    """Initialize the ClusterBasedReadabilityClassifier if not disabled."""
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
    """Initialize the LLMReadabilityClassifier if not disabled."""
    if args.disable_llm:
        return None
        
    logger.info("Initializing LLM readability classifier...")
    return LLMReadabilityClassifier(model_path=args.llm_model_path)

def init_jargon_classifier(args: argparse.Namespace) -> Optional[MedicalJargonClassifier]:
    """Initialize the MedicalJargonClassifier if not disabled."""
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
    
    # Filter out None values
    enabled_classifiers = {k: v for k, v in classifiers.items() if v is not None}
    successful = list(enabled_classifiers.keys())
    failed = [k for k, v in classifiers.items() if v is None]
    
    if successful:
        logger.info(f"Successfully initialized classifiers: {successful}")
    if failed:
        logger.warning(f"Failed to initialize classifiers: {failed}")
    
    return enabled_classifiers


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the simplified script."""
    parser = argparse.ArgumentParser(description='Calculate text readability metrics for response texts')

    # ===== GENERAL OPTIONS =====
    general = parser.add_argument_group('General Options')
    general.add_argument('--output', type=str, default='readability_metrics.json', help='Output file path')
    general.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')

    # ===== DATASET OPTIONS =====
    dataset = parser.add_argument_group('Dataset Options')
    dataset.add_argument('--dataset', type=str, 
                        choices=['baseline', 'fewshot', 'finetuned', 'finetuned3b', 'finetunedIt', 'claude', 'custom'],
                        default='baseline', help='Dataset to process')
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
    masked_prob_group.add_argument('--masked-prob-model', type=str, 
                                  default='emilyalsentzer/Bio_ClinicalBERT', 
                                  help='Model name for masked probability metric')
    masked_prob_group.add_argument('--masked-prob-method', type=str,
                                  choices=['random', 'np', 'rnptc'],
                                  default='rnptc',
                                  help='Method for masked probability: random, np (noun phrase), or rnptc (ranked NP)')
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


def get_input_config(args) -> tuple[str, str]:
    """Get input path and dataset name based on arguments."""
    if args.dataset != 'custom':
        return DATASET_PATHS[args.dataset], args.dataset
    
    if not args.custom_dataset_path:
        logger.error("Custom dataset path must be provided with --custom-dataset-path")
        raise ValueError("Custom dataset path required")
    
    input_path = args.custom_dataset_path
    dataset_name = Path(input_path).stem
    return input_path, dataset_name


def validate_input_file(input_path: str) -> None:
    """Validate that the input file exists."""
    if not Path(input_path).exists():
        logger.error(f"Input file {input_path} does not exist")
        raise FileNotFoundError(f"Input file {input_path} does not exist")
    
    
def main() -> None:
    """Execute the processing pipeline."""
    args = parse_arguments()
    config_dict = vars(args)
    
    # Initialize components
    classifiers = initialize_classifiers(args)
    scorer = TextReadabilityScorer(classifiers, config=config_dict)
    
    # Get input path and dataset name
    input_path, dataset_name = get_input_config(args)
    
    # Validate input file exists
    validate_input_file(input_path)
    
    # Process and save results
    results = scorer.process_file(input_path, args.batch_size, dataset_name)
    scorer.save_results(results, args.output)
    logger.info(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()