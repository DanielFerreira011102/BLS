#!/bin/bash
#SBATCH --job-name=bls_metrics    # Name of the job visible in queue
#SBATCH --nodes=1                  # Number of compute nodes to allocate
#SBATCH --ntasks=1                 # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12         # CPU cores per task
#SBATCH --mem=90G                  # Memory allocation per node - INCREASED
#SBATCH --gres=gpu:1               # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu            # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq1  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq1/logs/%j/logs.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq1/logs/%j/logs.err   # Standard error file (%j = job ID)

#===============================================================================
# Biomedical Language Simplification (BLS) - Dataset Metrics Calculation
# This script calculates readability metrics for simple and expert texts
#===============================================================================

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

# Display colorful log messages
log_section() {
    echo ""
    echo -e "\033[1;36m=== $1 ===\033[0m"
    echo ""
}

log_info() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

log_warn() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

log_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1" >&2
}

# Check if command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        log_error "$1"
        exit 1
    fi
}

# Show usage information
show_usage() {
    echo "Usage: sbatch run.sh --dataset NAME --metric NAME [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  --dataset NAME          Dataset to process (e.g., cochrane, claude, plaba-sentence, plaba-paragraph, custom)"
    echo "  --metric NAME           Metric to compute (e.g., textstat, syntax, umls, commonlit, masked_prob, gispy, scigispy, cluster, llm, jargon)"
    echo ""
    echo "Optional Arguments:"
    echo "  --output-base PATH      Base directory for output (default: \$OUTPUT_DIR/phase0)"
    echo "  --batch-size SIZE       Batch size for processing (default: 32)"
    echo "  --simple-column NAME    Column name for simple text (default: Simple)"
    echo "  --expert-column NAME    Column name for expert text (default: Expert)"
    echo "  --spacy-model-syntax MODEL spaCy model for syntax metric (default: en_core_web_trf)"
    echo "  --spacy-model-umls MODEL spaCy model for UMLS metric (default: en_core_sci_scibert)"
    echo "  --spacy-model-jargon MODEL spaCy model for jargon metric (default: en_core_web_trf)"
    echo "  --commonlit-model-type TYPE CommonLit model type (default: albert)"
    echo "  --masked-prob-model MODEL Model for masked probability (default: emilyalsentzer/Bio_ClinicalBERT)"
    echo "  --masked-prob-method METHOD Method for masked probability (default: rnptc)"
    echo "  --num-masks NUM           Number of masks to use for masked_prob metric (default: 10)"
    echo "  --mask-fraction FRAC      Fraction of text to mask for masked_prob metric (default: 0.15)"
    echo "  --use-corenlp           Use CoreNLP for GisPy metric (NOT recommended - much slower)"
    echo "  --corenlp-memory MEM    Memory allocation for CoreNLP (default: 4G)"
    echo "  --scigispy-window-size SIZE Window size for SciGisPy (default: 1)"
    echo "  --scigispy-breakpoint-percentile PERC Breakpoint percentile for SciGisPy (default: 75)"
    echo "  --llm-model-path PATH   Path to LLM model (default: /beegfs/client/default/dl-models/turbomind/deepseek-r1-distill-llama-70b-awq-TurboMind)"
    echo "  --cluster-model-path PATH Path to cluster model (default: /data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/clear)"
    echo "  --jargon-model-path PATH Path to jargon model (default: /data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model)"
    echo "  --jargon-pretrained-model MODEL Pretrained model for jargon classifier (default: roberta-large)"
    echo ""
    echo "Examples:"
    echo "  sbatch run.sh --dataset cochrane --metric textstat"
    echo "  sbatch run.sh --dataset claude --metric gispy"
    echo "  sbatch run.sh --dataset cochrane --metric jargon"
    exit 1
}

#===============================================================================
# DIRECTORY SETUP
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq1"
BASE_DIR="$HOME_DIR/storage/bls"
RESOURCE_DIR="$BASE_DIR/resources"
WORK_DIR="$BASE_DIR/rq1"
LOGS_DIR="$WORK_DIR/logs/$SLURM_JOB_ID"
OUTPUT_DIR="$WORK_DIR/outputs"
DATA_DIR="$RESOURCE_DIR/data"
NLP_DIR="$RESOURCE_DIR/nlp"

# Create required directories
log_info "Creating output directories..."
mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"
check_status "Failed to create output directories"

#===============================================================================
# ENVIRONMENT SETUP
#===============================================================================

log_section "Environment Setup"

# Load required system modules
log_info "Loading required modules..."
module load python
check_status "Failed to load Python module"

module load gcc/13.2.0-l6taobu
check_status "Failed to load GCC module"

module load openjdk
check_status "Failed to load OpenJDK module"

module load cuda/12.4.0-x4k27pl
check_status "Failed to load CUDA module"

# Set environment variables
CUDA_LIB_PATH="/apps/local/spack/spack/linux-ubuntu22.04-x86_64_v2/gcc-13.2.0/cuda-12.4.0-x4k27plupvaengzt57xrhk7mijgplgem/lib64"
export LD_LIBRARY_PATH="$(gcc -print-file-name=libstdc++.so.6 | xargs dirname):$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
export CUDA_VISIBLE_DEVICES=0
export NLTK_DATA="$NLP_DIR/WordNetNLTK:$NLP_DIR/PunktNLTK:$NLTK_DATA"
export NLTK_DATA="/data/home/djbf/storage/bls/resources/nlp/WordNetNLTK:/data/home/djbf/storage/bls/resources/nlp/PunktNLTK:$NLTK_DATA"

# Activate virtual environment
log_info "Activating virtual environment..."
source "$HOME_DIR/bls/venv/bin/activate"
check_status "Failed to activate virtual environment"

# Verify GPU availability
log_section "GPU Information"
nvidia-smi
check_status "Failed to query GPU information"

#===============================================================================
# ARGUMENT PARSING
#===============================================================================

log_section "Argument Parsing"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift 2 ;;
        --metric) METRIC="$2"; shift 2 ;;
        --output-base) OUTPUT_BASE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --simple-column) SIMPLE_COLUMN="$2"; shift 2 ;;
        --expert-column) EXPERT_COLUMN="$2"; shift 2 ;;
        --spacy-model-syntax) SPACY_MODEL_SYNTAX="$2"; shift 2 ;;
        --spacy-model-umls) SPACY_MODEL_UMLS="$2"; shift 2 ;;
        --spacy-model-jargon) SPACY_MODEL_JARGON="$2"; shift 2 ;;
        --commonlit-model-type) COMMONLIT_MODEL_TYPE="$2"; shift 2 ;;
        --masked-prob-model) MASKED_PROB_MODEL="$2"; shift 2 ;;
        --masked-prob-method) MASKED_PROB_METHOD="$2"; shift 2 ;;
        --num-masks) NUM_MASKS="$2"; shift 2 ;;
        --mask-fraction) MASK_FRACTION="$2"; shift 2 ;;
        --use-corenlp) USE_CORENLP=true; shift ;;
        --corenlp-memory) CORENLP_MEMORY="$2"; shift 2 ;;
        --scigispy-window-size) SCIGISPY_WINDOW_SIZE="$2"; shift 2 ;;
        --scigispy-breakpoint-percentile) SCIGISPY_BREAKPOINT_PERCENTILE="$2"; shift 2 ;;
        --llm-model-path) LLM_MODEL_PATH="$2"; shift 2 ;;
        --cluster-model-path) CLUSTER_MODEL_PATH="$2"; shift 2 ;;
        --jargon-model-path) JARGON_MODEL_PATH="$2"; shift 2 ;;
        --jargon-pretrained-model) JARGON_PRETRAINED_MODEL="$2"; shift 2 ;;
        --help|-h) show_usage ;;
        *) log_error "Unknown argument: $1"; show_usage ;;
    esac
done

# Set defaults
OUTPUT_BASE=${OUTPUT_BASE:-$OUTPUT_DIR/phase0}
BATCH_SIZE=${BATCH_SIZE:-128}
SIMPLE_COLUMN=${SIMPLE_COLUMN:-Simple}
EXPERT_COLUMN=${EXPERT_COLUMN:-Expert}
SPACY_MODEL_SYNTAX=${SPACY_MODEL_SYNTAX:-en_core_web_trf}
SPACY_MODEL_UMLS=${SPACY_MODEL_UMLS:-en_core_sci_scibert}
SPACY_MODEL_JARGON=${SPACY_MODEL_JARGON:-en_core_web_trf}
COMMONLIT_MODEL_TYPE=${COMMONLIT_MODEL_TYPE:-albert}
MASKED_PROB_MODEL=${MASKED_PROB_MODEL:-emilyalsentzer/Bio_ClinicalBERT}
MASKED_PROB_METHOD=${MASKED_PROB_METHOD:-rnptc}
NUM_MASKS=${NUM_MASKS:-10}
MASK_FRACTION=${MASK_FRACTION:-0.15}
USE_CORENLP=${USE_CORENLP:-false}
CORENLP_MEMORY=${CORENLP_MEMORY:-4G}
SCIGISPY_WINDOW_SIZE=${SCIGISPY_WINDOW_SIZE:-1}
SCIGISPY_BREAKPOINT_PERCENTILE=${SCIGISPY_BREAKPOINT_PERCENTILE:-75}
LLM_MODEL_PATH=${LLM_MODEL_PATH:-/beegfs/client/default/dl-models/turbomind/deepseek-r1-distill-llama-70b-awq-TurboMind}
CLUSTER_MODEL_PATH=${CLUSTER_MODEL_PATH:-/data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/clear}
JARGON_MODEL_PATH=${JARGON_MODEL_PATH:-/data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model}
JARGON_PRETRAINED_MODEL=${JARGON_PRETRAINED_MODEL:-roberta-large}

# Validate required arguments
if [ -z "$DATASET" ] || [ -z "$METRIC" ]; then
    log_error "Required arguments: --dataset and --metric"
    show_usage
fi

# Validate dataset
VALID_DATASETS=("cochrane" "claude" "plaba-sentence" "plaba-paragraph" "custom")
if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET} " ]]; then
    log_error "Invalid dataset: $DATASET. Must be one of: ${VALID_DATASETS[*]}"
    show_usage
fi

# Validate metric
VALID_METRICS=("textstat" "syntax" "umls" "commonlit" "masked_prob" "gispy" "scigispy" "cluster" "llm" "jargon")
if [[ ! " ${VALID_METRICS[*]} " =~ " ${METRIC} " ]]; then
    log_error "Invalid metric: $METRIC. Must be one of: ${VALID_METRICS[*]}"
    show_usage
fi

# Set CORENLP_HOME only if explicitly using CoreNLP
if [ "$USE_CORENLP" == "true" ]; then
    export CORENLP_HOME="$NLP_DIR/StanzaCoreNLP"
    log_warn "WARNING: Using CoreNLP for coreference resolution. This is significantly slower."
    log_warn "Consider using the default fastcoref implementation instead."
fi

#===============================================================================
# METRIC CONFIGURATION
#===============================================================================

log_section "Metric Configuration"

# List of all metrics
ALL_METRICS=(textstat syntax umls commonlit masked_prob gispy scigispy cluster llm jargon)

# Generate disable flags for all metrics except the selected one
DISABLE_FLAGS=""
for m in "${ALL_METRICS[@]}"; do
    if [ "$m" != "$METRIC" ]; then
        DISABLE_FLAGS+=" --disable-${m//_/-}"
    fi
done

# Configure metric-specific settings and output subdirectories
case "$METRIC" in
    textstat)
        SUBDIR="textstat"
        EXTRA_FLAGS=""
        ;;
    syntax)
        SUBDIR="syntax/$SPACY_MODEL_SYNTAX"
        EXTRA_FLAGS="--spacy-model-syntax $SPACY_MODEL_SYNTAX"
        ;;
    umls)
        SUBDIR="umls/$SPACY_MODEL_UMLS"
        EXTRA_FLAGS="--spacy-model-umls $SPACY_MODEL_UMLS"
        ;;
    commonlit)
        SUBDIR="commonlit/$COMMONLIT_MODEL_TYPE"
        EXTRA_FLAGS="--commonlit-model-type $COMMONLIT_MODEL_TYPE"
        ;;
    masked_prob)
        MASKED_PROB_MODEL_NAME=$(basename "$MASKED_PROB_MODEL")
        SUBDIR="masked_prob/$MASKED_PROB_METHOD/$MASKED_PROB_MODEL_NAME"
        EXTRA_FLAGS="--masked-prob-model $MASKED_PROB_MODEL --masked-prob-method $MASKED_PROB_METHOD --num-masks $NUM_MASKS --mask-fraction $MASK_FRACTION"
        ;;
    gispy)
        SUBDIR="gispy"
        if [ "$USE_CORENLP" == "true" ]; then
            SUBDIR="gispy/corenlp"
            EXTRA_FLAGS="--use-corenlp --corenlp-memory $CORENLP_MEMORY"
        else
            EXTRA_FLAGS=""
        fi
        ;;
    scigispy)
        if [ "$SCIGISPY_WINDOW_SIZE" != "1" ] || [ "$SCIGISPY_BREAKPOINT_PERCENTILE" != "75" ]; then
            SUBDIR="scigispy/ws${SCIGISPY_WINDOW_SIZE}_bp${SCIGISPY_BREAKPOINT_PERCENTILE}"
        else
            SUBDIR="scigispy"
        fi
        EXTRA_FLAGS="--scigispy-window-size $SCIGISPY_WINDOW_SIZE --scigispy-breakpoint-percentile $SCIGISPY_BREAKPOINT_PERCENTILE"
        ;;
    cluster)
        CLUSTER_MODEL_NAME=$(basename "$CLUSTER_MODEL_PATH")
        SUBDIR="cluster/$CLUSTER_MODEL_NAME"
        EXTRA_FLAGS="--cluster-model-path $CLUSTER_MODEL_PATH"
        ;;
    llm)
        LLM_MODEL_NAME=$(basename "$LLM_MODEL_PATH")
        SUBDIR="llm/$LLM_MODEL_NAME"
        EXTRA_FLAGS="--llm-model-path $LLM_MODEL_PATH"
        ;;
    jargon)
        JARGON_MODEL_NAME=$(basename "$JARGON_MODEL_PATH")
        JARGON_PRETRAINED_MODEL_NAME=$(basename "$JARGON_PRETRAINED_MODEL")
        SUBDIR="jargon/$JARGON_PRETRAINED_MODEL_NAME"
        EXTRA_FLAGS="--jargon-model-path $JARGON_MODEL_PATH --jargon-pretrained-model $JARGON_PRETRAINED_MODEL --spacy-model-jargon $SPACY_MODEL_JARGON"
        ;;
    *)
        log_error "Unknown metric: $METRIC"
        exit 1
        ;;
esac

# Define output path
OUTPUT_PATH="$OUTPUT_BASE/$DATASET/$SUBDIR/readability_metrics.json"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_PATH")"

log_info "Output path: $OUTPUT_PATH"

#===============================================================================
# EXECUTION
#===============================================================================

log_section "Starting Job Execution"

log_info "Running Python script with dataset: $DATASET and metric: $METRIC"

# Construct the Python command with additional parameters
PYTHON_CMD="python $SOURCE_DIR/metrics/phase0.py --dataset $DATASET --output $OUTPUT_PATH $DISABLE_FLAGS $EXTRA_FLAGS --batch-size $BATCH_SIZE --simple-column $SIMPLE_COLUMN --expert-column $EXPERT_COLUMN"

# Add explicit garbage collection
log_info "Adding explicit garbage collection module"
PYTHON_CMD="python -c \"import gc; gc.enable();\" && $PYTHON_CMD"

# Run the command
log_info "Executing: $PYTHON_CMD"
eval "$PYTHON_CMD"
execution_status=$?

if [ $execution_status -eq 0 ]; then
    log_section "Job Completed Successfully"
    log_info "Metrics saved to: $OUTPUT_PATH"
else
    log_section "Job Failed"
    log_error "Python script exited with status code: $execution_status"
    log_error "Check logs for more details: $LOGS_DIR/logs.err"
    exit $execution_status
fi

# Deactivate the virtual environment
deactivate

log_info "Job ID: $SLURM_JOB_ID completed at $(date)"

#===============================================================================
# USAGE EXAMPLES (For reference only, not executed)
#===============================================================================

# Basic usage - TextStat metrics:
#   SLURM:  sbatch run.sh --dataset claude --metric textstat
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/textstat/readability_metrics.json \
#           --disable-syntax --disable-umls --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --batch-size 32

# Syntax metric with custom model:
#   SLURM:  sbatch run.sh --dataset claude --metric syntax --syntax-model en_core_web_trf
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/syntax/en_core_web_trf/readability_metrics.json \
#           --disable-textstat --disable-umls --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --spacy-model-syntax en_core_web_trf --batch-size 32

# UMLS metric with scientific BERT model:
#   SLURM:  sbatch run.sh --dataset claude --metric umls --umls-model en_core_sci_scibert
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/umls/en_core_sci_scibert/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --spacy-model-umls en_core_sci_scibert --batch-size 32

# UMLS metric with web transformer model:
#   SLURM:  sbatch run.sh --dataset claude --metric umls --umls-model en_core_web_trf
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/umls/en_core_web_trf/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --spacy-model-umls en_core_web_trf --batch-size 32

# CommonLit with DeBERTa ensemble:
#   SLURM:  sbatch run.sh --dataset claude --metric commonlit --commonlit-model-type deberta_ensemble
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/commonlit/deberta/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --commonlit-model-type deberta_ensemble \
#           --commonlit-model-path /data/home/djbf/storage/bls/resources/models/deberta \
#           --batch-size 32

# CommonLit with ALBERT model:
#   SLURM:  sbatch run.sh --dataset claude --metric commonlit --commonlit-model-type albertinit_g
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/masked-prob/scibert_scivocab_uncased/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model allenai/scibert_scivocab_uncased --batch-size 32

# Masked probability with BERT base:
#   SLURM:  sbatch run.sh --dataset claude --metric masked-prob --masked-prob-model "google-bert/bert-base-uncased"
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/masked-prob/bert-base-uncased/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model google-bert/bert-base-uncased --batch-size 32

# Masked probability with BiomedNLP-BiomedBERT:
#   SLURM:  sbatch run.sh --dataset claude --metric masked-prob --masked-prob-model "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/masked-prob/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --batch-size 32

# Masked probability with BlueBERT:
#   SLURM:  sbatch run.sh --dataset claude --metric masked-prob --masked-prob-model "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/masked-prob/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --batch-size 32

# GisPy with FastCoREF:
#   SLURM:  sbatch run.sh --dataset claude --metric gispy
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/gispy/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --batch-size 32

# SciGisPy with custom parameters:
#   SLURM:  sbatch run.sh --dataset claude --metric scigispy --scigispy-window-size 3 --scigispy-breakpoint-percentile 10
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/scigispy/3/10/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-cluster --disable-llm --disable-jargon \
#           --scigispy-window-size 3 --scigispy-breakpoint-percentile 10 --batch-size 32

# Cluster metric (simple):
#   SLURM:  sbatch run.sh --dataset claude --metric cluster
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/cluster/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-llm --disable-jargon \
#           --batch-size 32

# LLM with default deepseek model:
#   SLURM:  sbatch run.sh --dataset claude --metric llm --batch-size 128
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/llm/deepseek-r1-distill-llama-70b-awq-TurboMind/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-jargon \
#           --batch-size 128 \
#           --llm-model-path /beegfs/client/default/dl-models/turbomind/deepseek-r1-distill-llama-70b-awq-TurboMind

# LLM with Nvidia Llama 3.1 model:
#   SLURM:  sbatch run.sh --dataset claude --metric llm --llm-model-path "/beegfs/client/default/dl-models/turbomind/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind"
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-jargon \
#           --batch-size 128 \
#           --llm-model-path /beegfs/client/default/dl-models/turbomind/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind

# LLM with OpenBioLLM model:
#   SLURM:  sbatch run.sh --dataset claude --metric llm --llm-model-path "/beegfs/client/default/dl-models/turbomind/Llama3-OpenBioLLM-70B-AWQ-INT4-TurboMind"
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/llm/Llama3-OpenBioLLM-70B-AWQ-INT4-TurboMind/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-jargon \
#           --batch-size 128 \
#           --llm-model-path /beegfs/client/default/dl-models/turbomind/Llama3-OpenBioLLM-70B-AWQ-INT4-TurboMind

# Medical jargon metrics:
#   SLURM:  sbatch run.sh --dataset claude --metric jargon
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/jargon/roberta-large/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-llm \
#           --batch-size 32 \
#           --jargon-model-path /data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model \
#           --jargon-pretrained-model roberta-large \
#           --spacy-model-jargon en_core_web_trf

# Medical jargon with different spaCy model:
#   SLURM:  sbatch run.sh --dataset claude --metric jargon --spacy-model-jargon en_core_web_trf
#   PYTHON: python -m metrics.phase0 --dataset claude \
#           --output $OUTPUT_DIR/phase0/claude/jargon/roberta-large/readability_metrics.json \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-llm \
#           --batch-size 32 \
#           --jargon-model-path /data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model \
#           --jargon-pretrained-model roberta-large \
#           --spacy-model-jargon en_core_sci_scibert