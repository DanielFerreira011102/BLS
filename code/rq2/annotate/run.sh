#!/bin/bash
#SBATCH --job-name=bls_annotate    # Name of the job visible in queue
#SBATCH --nodes=1                  # Number of compute nodes to allocate
#SBATCH --ntasks=1                 # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12         # CPU cores per task
#SBATCH --mem=90G                  # Memory allocation per node - INCREASED
#SBATCH --gres=gpu:1               # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu            # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq2  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq2/logs/%j/logs.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq2/logs/%j/logs.err   # Standard error file (%j = job ID)

#===============================================================================
# Biomedical Language Simplification (BLS) - Dataset Annotation
# This script annotates answer variants with readability metrics
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
    echo "Usage: sbatch $0 --dataset NAME --metric NAME [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  --dataset NAME          Dataset to process (e.g., liveqa, medicationqa, bioasq)"
    echo "  --metric NAME           Metric to compute (e.g., textstat, syntax, umls, commonlit, masked_prob, gispy, scigispy, cluster, llm, jargon)"
    echo ""
    echo "Optional Arguments:"
    echo "  --output-base PATH      Base directory for output (default: \$OUTPUT_DIR/phase1)"
    echo "  --batch-size SIZE       Batch size for processing (default: 32)"
    echo "  --checkpoint-path PATH  Path to checkpoint file (default: based on output path)"
    echo "  --checkpoint-interval INT Number of entries between checkpoints (default: 1000)"
    echo "  --spacy-model-syntax MODEL spaCy model for syntax metric (default: en_core_web_trf)"
    echo "  --spacy-model-umls MODEL spaCy model for UMLS metric (default: en_core_sci_scibert)"
    echo "  --spacy-model-jargon MODEL spaCy model for jargon metric (default: en_core_web_trf)"
    echo "  --commonlit-model-type TYPE CommonLit model type (default: albert)"
    echo "  --masked-prob-model MODEL Model for masked probability (default: emilyalsentzer/Bio_ClinicalBERT)"
    echo "  --masked-prob-method METHOD Method for masked probability: random, np, or rnptc (default: random)"
    echo "  --num-masks NUM           Number of masks to use for masked_prob metric (default: 10)"
    echo "  --mask-fraction FRAC      Fraction of text to mask for masked_prob metric (default: 0.15)"
    echo "  --use-corenlp           Use CoreNLP for GisPy metric (NOT recommended - much slower)"
    echo "  --corenlp-memory MEM    Memory allocation for CoreNLP (default: 4G)"
    echo "  --scigispy-window-size SIZE Window size for SciGisPy (default: 1)"
    echo "  --scigispy-breakpoint-percentile PERC Breakpoint percentile for SciGisPy (default: 75)"
    echo "  --llm-model-path PATH   Path to LLM model (default: /data/home/djbf/storage/bls/rq2/outputs/llmify/default/improved/mse_1.0+kl_0.0+ce_0.0+soft_0.0)"
    echo "  --cluster-model-path PATH Path to cluster model (default: /data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/medreadme)"
    echo "  --jargon-model-path PATH Path to jargon model (default: /data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model)"
    echo "  --jargon-pretrained-model MODEL Pretrained model for jargon classifier (default: roberta-large)"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 --dataset liveqa --metric textstat"
    echo "  sbatch $0 --dataset liveqa --metric gispy"
    echo "  sbatch $0 --dataset liveqa --metric masked_prob --masked-prob-method rnptc"
    echo "  sbatch $0 --dataset liveqa --metric jargon"
    exit 1
}

#===============================================================================
# DIRECTORY SETUP
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq2"
BASE_DIR="$HOME_DIR/storage/bls"
RESOURCE_DIR="$BASE_DIR/resources"
WORK_DIR="$BASE_DIR/rq2"
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
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        --checkpoint-interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
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
OUTPUT_BASE=${OUTPUT_BASE:-$OUTPUT_DIR/phase1}
BATCH_SIZE=${BATCH_SIZE:-128}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-100}
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
LLM_MODEL_PATH=${LLM_MODEL_PATH:-/data/home/djbf/storage/bls/rq2/outputs/llmify/default/improved/mse_1.0+kl_0.0+ce_0.0+soft_0.0}
CLUSTER_MODEL_PATH=${CLUSTER_MODEL_PATH:-/data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/medreadme}
JARGON_MODEL_PATH=${JARGON_MODEL_PATH:-/data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model}
JARGON_PRETRAINED_MODEL=${JARGON_PRETRAINED_MODEL:-roberta-large}

# Validate required arguments
if [ -z "$DATASET" ] || [ -z "$METRIC" ]; then
    log_error "Required arguments: --dataset and --metric"
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
            SUBDIR="gispy"
            EXTRA_FLAGS="--use-corenlp --corenlp-memory $CORENLP_MEMORY"
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

# Define output and checkpoint paths
OUTPUT_PATH="$OUTPUT_BASE/$DATASET/$SUBDIR/readability_metrics.json"
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$OUTPUT_BASE/$DATASET/$SUBDIR/checkpoint.json}

# Ensure directories exist
mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$CHECKPOINT_PATH")"

log_info "Output path: $OUTPUT_PATH"
log_info "Checkpoint path: $CHECKPOINT_PATH"

#===============================================================================
# EXECUTION
#===============================================================================

log_section "Starting Job Execution"

log_info "Running Python script with dataset: $DATASET and metric: $METRIC"

# Construct the Python command with additional parameters
PYTHON_CMD="python $SOURCE_DIR/annotate/phase1.py --dataset $DATASET --output $OUTPUT_PATH --checkpoint-path $CHECKPOINT_PATH --checkpoint-interval $CHECKPOINT_INTERVAL $DISABLE_FLAGS $EXTRA_FLAGS --batch-size $BATCH_SIZE"

# Add explicit garbage collection
log_info "Adding explicit garbage collection module"
PYTHON_CMD="python -c \"import gc; gc.enable();\" && $PYTHON_CMD"

# Run the command
log_info "Executing: $PYTHON_CMD"
eval "$PYTHON_CMD"
execution_status=$?

if [ $execution_status -eq 0 ]; then
    log_section "Job Completed Successfully"
    log_info "Annotated readability metrics saved to: $OUTPUT_PATH"
    log_info "Checkpoint saved to: $CHECKPOINT_PATH"
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
#   SLURM:  sbatch run.sh --dataset liveqa --metric textstat
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/textstat/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/textstat/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-syntax --disable-umls --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --batch-size 128

# Syntax metric with custom model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric syntax --spacy-model-syntax en_core_web_trf
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/syntax/en_core_web_trf/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/syntax/en_core_web_trf/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-umls --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --spacy-model-syntax en_core_web_trf --batch-size 128

# UMLS metric with scientific BERT model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric umls --spacy-model-umls en_core_sci_scibert
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/umls/en_core_sci_scibert/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/umls/en_core_sci_scibert/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --spacy-model-umls en_core_sci_scibert --batch-size 128

# UMLS metric with web transformer model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric umls --spacy-model-umls en_core_web_trf
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/umls/en_core_web_trf/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/umls/en_core_web_trf/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-commonlit --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --spacy-model-umls en_core_web_trf --batch-size 128

# CommonLit with DeBERTa ensemble:
#   SLURM:  sbatch run.sh --dataset liveqa --metric commonlit --commonlit-model-type deberta_ensemble
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/commonlit/deberta_ensemble/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/commonlit/deberta_ensemble/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --commonlit-model-type deberta_ensemble \
#           --batch-size 128

# CommonLit with ALBERT model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric commonlit --commonlit-model-type albert
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/commonlit/albert/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/commonlit/albert/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-masked-prob \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --commonlit-model-type albert \
#           --batch-size 128

# Masked probability with Bio_ClinicalBERT:
#   SLURM:  sbatch run.sh --dataset liveqa --metric masked_prob --masked-prob-model "emilyalsentzer/Bio_ClinicalBERT"
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/Bio_ClinicalBERT/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/Bio_ClinicalBERT/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model emilyalsentzer/Bio_ClinicalBERT --masked-prob-method rnptc --batch-size 128

# Masked probability with SciBERT:
#   SLURM:  sbatch run.sh --dataset liveqa --metric masked_prob --masked-prob-model "allenai/scibert_scivocab_uncased"
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/scibert_scivocab_uncased/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/scibert_scivocab_uncased/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model allenai/scibert_scivocab_uncased --masked-prob-method rnptc --batch-size 128

# Masked probability with BERT base:
#   SLURM:  sbatch run.sh --dataset liveqa --metric masked_prob --masked-prob-model "google-bert/bert-base-uncased"
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/bert-base-uncased/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/bert-base-uncased/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model google-bert/bert-base-uncased --masked-prob-method rnptc --batch-size 128

# Masked probability with BiomedNLP-BiomedBERT:
#   SLURM:  sbatch run.sh --dataset liveqa --metric masked_prob --masked-prob-model "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --masked-prob-method rnptc --batch-size 128

# Masked probability with BlueBERT:
#   SLURM:  sbatch run.sh --dataset liveqa --metric masked_prob --masked-prob-model "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/masked_prob/rnptc/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-gispy --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --masked-prob-model bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12 --masked-prob-method rnptc --batch-size 128

# GisPy with FastCoref (default):
#   SLURM:  sbatch run.sh --dataset liveqa --metric gispy
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/gispy/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/gispy/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --batch-size 128

# GisPy with CoreNLP (slower):
#   SLURM:  sbatch run.sh --dataset liveqa --metric gispy --use-corenlp --corenlp-memory 6G
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/gispy/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/gispy/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-scigispy --disable-cluster --disable-llm --disable-jargon \
#           --use-corenlp --corenlp-memory 6G --batch-size 128

# SciGisPy with custom parameters:
#   SLURM:  sbatch run.sh --dataset liveqa --metric scigispy --scigispy-window-size 3 --scigispy-breakpoint-percentile 10
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/scigispy/ws3_bp10/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/scigispy/ws3_bp10/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-cluster --disable-llm --disable-jargon \
#           --scigispy-window-size 3 --scigispy-breakpoint-percentile 10 --batch-size 128

# Cluster metric with default model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric cluster
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/cluster/clear/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/cluster/clear/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-llm --disable-jargon \
#           --cluster-model-path /data/home/djbf/storage/bls/resources/models/cluster-readability-model-svr-300/clear \
#           --batch-size 128

# LLM with default model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric llm
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/llm/mse_1.0+kl_0.0+ce_0.0+soft_0.0/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/llm/mse_1.0+kl_0.0+ce_0.0+soft_0.0/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-jargon \
#           --llm-model-path /data/home/djbf/storage/bls/rq2/outputs/llmify/default/improved/mse_1.0+kl_0.0+ce_0.0+soft_0.0 \
#           --batch-size 128

# Medical jargon metrics:
#   SLURM:  sbatch run.sh --dataset liveqa --metric jargon
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/jargon/roberta-large/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/jargon/roberta-large/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-llm \
#           --jargon-model-path /data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model \
#           --jargon-pretrained-model roberta-large \
#           --spacy-model-jargon en_core_web_trf \
#           --batch-size 128

# Medical jargon with scientific BERT model:
#   SLURM:  sbatch run.sh --dataset liveqa --metric jargon --spacy-model-jargon en_core_web_trf
#   PYTHON: python -m annotate.phase1 --dataset liveqa \
#           --output $OUTPUT_DIR/phase1/liveqa/jargon/roberta-large/readability_metrics.json \
#           --checkpoint-path $OUTPUT_DIR/phase1/liveqa/jargon/roberta-large/checkpoint.json \
#           --checkpoint-interval 100 \
#           --disable-textstat --disable-syntax --disable-umls --disable-commonlit \
#           --disable-masked-prob --disable-gispy --disable-scigispy --disable-cluster --disable-llm \
#           --jargon-model-path /data/home/djbf/storage/bls/resources/models/jargon-readability-model-crf/medreadme/best_model \
#           --jargon-pretrained-model roberta-large \
#           --spacy-model-jargon en_core_web_trf \
#           --batch-size 128