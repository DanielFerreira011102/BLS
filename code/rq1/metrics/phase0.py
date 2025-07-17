import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from datasets import load_dataset
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

from utils.helpers import load_json, save_json, setup_logging

# Set up logging
logger = setup_logging()

# Constants for metric types
BATCH_METRICS = {'commonlit', 'syntax', 'textstat', 'cluster', 'llm', 'jargon'}
SINGLE_METRICS = {'gispy', 'scigispy', 'masked_prob', 'umls'}

class TextReadabilityScorer:
    """Calculates readability metrics on texts using various classifiers."""

    def __init__(self, classifiers: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the scorer with classifiers and configuration."""
        self.classifiers = classifiers
        self.enabled_metrics = set(classifiers.keys())
        self.config = config or {}
        logger.info(f"Enabled metric groups: {self.enabled_metrics}")

    def calculate_metrics_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Calculates all enabled metrics for a batch of texts."""
        results = [{metric: {} for metric in self.enabled_metrics} for _ in range(len(texts))]

        # Process batch metrics
        for metric in BATCH_METRICS & self.enabled_metrics:
            classifier = self.classifiers[metric]
            scores = classifier.predict_batch(texts, batch_size=batch_size)
            for i, score_data in enumerate(scores):
                results[i][metric] = score_data

        # Process single metrics in batches
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Processing single metrics"):
            batch_end = min(batch_start + batch_size, len(texts))
            for i in range(batch_start, batch_end):
                text = texts[i]
                for metric in SINGLE_METRICS & self.enabled_metrics:
                    classifier = self.classifiers[metric]
                    results[i][metric] = classifier.predict_single(text)

        return results

    def process_texts(self, df: pd.DataFrame, text_columns: Dict[str, str], dataset_name: str, batch_size: int = 32) -> Dict[str, Any]:
        """Processes all texts in a dataset for both simple and expert versions."""
        logger.info(f"Processing dataset: {dataset_name}")

        # Validate columns
        for version in ['simple', 'expert']:
            column = text_columns[version]
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in dataset '{dataset_name}'")

        # Initialize results
        results = [{'simple': {'text': None, 'metrics': {}}, 'expert': {'text': None, 'metrics': {}}} for _ in range(len(df))]

        # Process each version
        for version in ['simple', 'expert']:
            logger.info(f"Processing {version} texts")
            texts = df[text_columns[version]].tolist()
            batch_metrics = self.calculate_metrics_batch(texts, batch_size)
            for pos, metrics in enumerate(batch_metrics):
                results[pos][version]['text'] = texts[pos]
                results[pos][version]['metrics'] = metrics

        return {'dataset': dataset_name, 'config': self.config, 'samples': results}

    def process_datasets(self, datasets: Dict[str, tuple[pd.DataFrame, Dict[str, str]]], batch_size: int = 32) -> Dict[str, Any]:
        """Processes multiple datasets."""
        results = {}
        for name, (df, columns) in datasets.items():
            results[name] = self.process_texts(df, columns, name, batch_size)
        return results

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Saves results to a JSON file."""
        save_json(results, output_path)

# Dataset loading functions
def load_cochrane_dataset(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Loads the Cochrane dataset from Hugging Face."""
    logger.info("Loading Cochrane dataset from Hugging Face")
    dataset = load_dataset('GEM/cochrane-simplification', trust_remote_code=True)
    df = pd.concat([dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()], ignore_index=True)
    df.rename(columns={'source': 'Expert', 'target': 'Simple'}, inplace=True)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    logger.info(f"Loaded {len(df)} question-answer pairs from Cochrane dataset")
    return df


def load_claude_dataset(sample_size: Optional[int] = None) -> pd.DataFrame:
    """Loads the Claude dataset from a local JSON file."""
    logger.info("Loading Claude dataset from local JSON file")
    dataset = load_json("/data/home/djbf/storage/bls/resources/datasets/claude/health_qa_20250222_205257.json")
    df = pd.DataFrame(dataset.get('qa_pairs', []))
    df.rename(columns={'question': 'Question', 'simple_answer': 'Simple', 'expert_answer': 'Expert'}, inplace=True)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    logger.info(f"Loaded {len(df)} question-answer pairs from Claude dataset")
    return df


def load_plaba_dataset(by: str = "sentence", sample_size: Optional[int] = None) -> pd.DataFrame:
    """Loads the Plaba dataset from a local CSV file."""
    logger.info(f"Loading Plaba ({by}) dataset from local CSV file")
    df = pd.read_csv(f"/data/home/djbf/storage/bls/resources/datasets/plaba/plaba_{by}.csv")
    df.rename(columns={'Question': 'Question', 'Simplification1': 'Simple', 'Expert': 'Expert'}, inplace=True)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
    
    logger.info(f"Loaded {len(df)} question-answer pairs from Plaba ({by}) dataset")
    return df

def load_custom_dataset(path: str, format: str = 'csv') -> pd.DataFrame:
    """Loads a custom dataset from a file."""
    path = Path(path)
    
    if format.lower() == 'csv':
        return pd.read_csv(path)
    elif format.lower() == 'json':
        return pd.read_json(path)
    elif format.lower() in ('excel', 'xlsx'):
        return pd.read_excel(path)
    
    raise ValueError(f"Unsupported format: {format}")

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
    """Initializes all classifiers based on command-line arguments."""
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
    
    successful = [metric for metric, classifier in classifiers.items() if classifier is not None]
    failed = [metric for metric, classifier in classifiers.items() if classifier is None]
    
    if successful:
        logger.info(f"Successfully initialized classifiers: {successful}")
    
    if failed:
        logger.warning(f"Failed to initialize classifiers: {failed}")
    
    return classifiers


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments with improved organization."""
    parser = argparse.ArgumentParser(description='Calculate text readability metrics')
    
    # ===== GENERAL OPTIONS =====
    general_group = parser.add_argument_group('General Options')
    general_group.add_argument('--batch-size', type=int, default=32, 
                               help='Batch size for processing')
    general_group.add_argument('--output', type=str, default='readability_metrics.json', 
                               help='Output file path')
    
    # ===== DATASET OPTIONS =====
    dataset_group = parser.add_argument_group('Dataset Options')
    dataset_group.add_argument('--dataset', type=str, 
                               choices=['cochrane', 'claude', 'plaba-sentence', 'plaba-paragraph', 'custom'], 
                               default='cochrane',
                               help='Dataset to process')
    dataset_group.add_argument('--custom-dataset-path', type=str, 
                               help='Path to custom dataset')
    dataset_group.add_argument('--custom-dataset-format', type=str, default='csv', 
                               choices=['csv', 'json', 'excel', 'xlsx'],
                               help='Format of custom dataset')
    dataset_group.add_argument('--simple-column', type=str, default='Simple', 
                               help='Column name for simple text')
    dataset_group.add_argument('--expert-column', type=str, default='Expert', 
                               help='Column name for expert text')
    
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
    # Feature toggles
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
    # Resources
    gispy_group.add_argument('--megahr-path', type=str, 
                            default='/data/home/djbf/storage/bls/resources/datasets/megahr/megahr.en.sort.i.txt', 
                            help='Path to MegaHR dictionary file')
    gispy_group.add_argument('--mrc-path', type=str, 
                            default='/data/home/djbf/storage/bls/resources/datasets/mrc/mrc_psycholinguistic_database.csv', 
                            help='Path to MRC database file')
    
    # ===== SCIGISPY OPTIONS =====
    scigispy_group = parser.add_argument_group('SciGisPy Options')
    # Feature toggles
    scigispy_group.add_argument('--disable-scigispy-hypernymy', action='store_true', 
                               help='Disable hypernymy metrics')
    scigispy_group.add_argument('--disable-scigispy-verb-overlap', action='store_true', 
                               help='Disable verb overlap metrics')
    scigispy_group.add_argument('--disable-scigispy-cohesion', action='store_true', 
                               help='Disable cohesion metrics')
    scigispy_group.add_argument('--disable-scigispy-ic', action='store_true', 
                               help='Disable information content metrics')
    # Resources
    scigispy_group.add_argument('--fasttext-path', type=str, 
                               default='/data/home/djbf/storage/bls/resources/models/crawl-300d-2M-subword/crawl-300d-2M-subword.bin', 
                               help='Path to FastText embedding model')
    scigispy_group.add_argument('--biowordvec-path', type=str, 
                               default='/data/home/djbf/storage/bls/resources/models/BioWordVec/BioWordVec_PubMed_MIMICIII_d200.bin', 
                               help='Path to BioWordVec embedding model')
    scigispy_group.add_argument('--corpus-stats-dir', type=str, 
                               default='/data/home/djbf/storage/bls/resources/datasets/ic/corpus_stats_without_pubmed', 
                               help='Path to corpus statistics directory')
    # Chunking configuration
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
                          default='/beegfs/client/default/dl-models/turbomind/deepseek-r1-distill-llama-70b-awq-TurboMind', 
                          help='Path to LLM model for readability estimation')
    
    # ===== MEDICAL JARGON OPTIONS =====
    jargon_group = parser.add_argument_group('Medical Jargon Options')
    jargon_group.add_argument('--jargon-pretrained-model', type=str, default='roberta-large', 
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
    """Main function to process datasets and calculate metrics."""
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
    if args.dataset == 'cochrane':
        df = load_cochrane_dataset()
    elif args.dataset == 'claude':
        df = load_claude_dataset()
    elif args.dataset == 'plaba-sentence':
        df = load_plaba_dataset(by="sentence")
    elif args.dataset == 'plaba-paragraph':
        df = load_plaba_dataset(by="paragraph")
    else:
        if not args.custom_dataset_path:
            raise ValueError("Custom dataset path must be provided with --custom-dataset-path")
        df = load_custom_dataset(args.custom_dataset_path, args.custom_dataset_format)

    # Define column mapping
    text_columns = {'simple': args.simple_column, 'expert': args.expert_column}

    # Process dataset
    datasets = {args.dataset: (df, text_columns)}
    results = scorer.process_datasets(datasets, args.batch_size)

    # Save results
    scorer.save_results(results, args.output)
    logger.info(f"Processing complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()