#!/bin/bash
#SBATCH --job-name=bls_variants    # Name of the job visible in queue
#SBATCH --nodes=1                  # Number of compute nodes to allocate
#SBATCH --ntasks=1                 # Number of tasks (processes) to create
#SBATCH --cpus-per-task=24         # CPU cores per task
#SBATCH --mem=80G                  # Memory allocation per node
#SBATCH --gres=gpu:1               # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu            # Compute partition/queue to use
#SBATCH --nodelist=gpu-srv-03      # Specific node to run on
#SBATCH --chdir=/data/home/djbf/storage/bls/rq2  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq2/logs/%j/logs.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq2/logs/%j/logs.err   # Standard error file (%j = job ID)

#===============================================================================
# Biomedical Language Simplification (BLS) - Variant Generation
# This script generates multiple answer variants for biomedical questions
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
    echo "Usage: sbatch $0 --dataset NAME --num-variants NUM"
    echo ""
    echo "Required Arguments:"
    echo "  --dataset NAME          Dataset to process (e.g., liveqa, medicationqa, bioasq)"
    echo "  --num-variants NUM      Number of variants to generate per question"
    echo ""
    echo "Optional Arguments:"
    echo "  --batch-size SIZE       Batch size for processing (default: 128)"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 --dataset liveqa --num-variants 5"
    echo "  sbatch $0 --dataset medicationqa --num-variants 10 --batch-size 64"
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
export CORENLP_HOME="$NLP_DIR/StanzaCoreNLP:$CORENLP_HOME"
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
        --num-variants) NUM_VARIANTS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --help|-h) show_usage ;;
        *) log_error "Unknown argument: $1"; show_usage ;;
    esac
done

# Set defaults and validate
DATASET=${DATASET:-liveqa}
NUM_VARIANTS=${NUM_VARIANTS:-5}
BATCH_SIZE=${BATCH_SIZE:-128}

if ! [[ "$NUM_VARIANTS" =~ ^[0-9]+$ ]] || [ "$NUM_VARIANTS" -lt 1 ]; then
    log_error "Number of variants must be a positive integer"
    show_usage
fi

# Validate required arguments
if [ -z "$DATASET" ]; then
    log_error "Required argument: --dataset"
    show_usage
fi

#===============================================================================
# OUTPUT CONFIGURATION
#===============================================================================

log_section "Output Configuration"

# Define output paths
OUTPUT_PATH="$OUTPUT_DIR/phase0/elo/$DATASET/answer_variants.json"
CHECKPOINT_PATH="$OUTPUT_DIR/phase0/elo/$DATASET/checkpoint.json"

# Ensure directories exist
mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$CHECKPOINT_PATH")"

log_info "Output path: $OUTPUT_PATH"
log_info "Checkpoint path: $CHECKPOINT_PATH"

#===============================================================================
# EXECUTION
#===============================================================================

log_section "Starting Job Execution"

log_info "Running Python script with dataset: $DATASET and num-variants: $NUM_VARIANTS"

# Construct the Python command
PYTHON_CMD="python $SOURCE_DIR/generation/phase0.py --use-elo --dataset $DATASET --output $OUTPUT_PATH --checkpoint-path $CHECKPOINT_PATH --num-variants $NUM_VARIANTS --batch-size $BATCH_SIZE"

# Run the command
log_info "Executing: $PYTHON_CMD"
eval "$PYTHON_CMD"
execution_status=$?

if [ $execution_status -eq 0 ]; then
    log_section "Job Completed Successfully"
    log_info "Generated answer variants saved to: $OUTPUT_PATH"
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

# Generate 5 variants for LiveQA dataset (default):
#   SLURM:  sbatch $0 --dataset liveqa --num-variants 5
#   PYTHON: python -m generation.phase0 --use-elo \
#           --dataset liveqa \
#           --output $OUTPUT_DIR/phase0/elo/liveqa/answer_variants.json \
#           --checkpoint-path $OUTPUT_DIR/phase0/elo/liveqa/checkpoint.json \
#           --num-variants 5 \
#           --batch-size 128

# Generate 10 variants for MedicationQA dataset:
#   SLURM:  sbatch $0 --dataset medicationqa --num-variants 10
#   PYTHON: python -m generation.phase0 --use-elo \
#           --dataset medicationqa \
#           --output $OUTPUT_DIR/phase0/elo/medicationqa/answer_variants.json \
#           --checkpoint-path $OUTPUT_DIR/phase0/elo/medicationqa/checkpoint.json \
#           --num-variants 10 \
#           --batch-size 128

# Generate 3 variants for MedQuAD dataset:
#   SLURM:  sbatch $0 --dataset medquad --num-variants 3
#   PYTHON: python -m generation.phase0 --use-elo \
#           --dataset medquad \
#           --output $OUTPUT_DIR/phase0/elo/medquad/answer_variants.json \
#           --checkpoint-path $OUTPUT_DIR/phase0/elo/medquad/checkpoint.json \
#           --num-variants 3 \
#           --batch-size 128

# Generate 7 variants for BioASQ dataset:
#   SLURM:  sbatch $0 --dataset bioasq --num-variants 7
#   PYTHON: python -m generation.phase0 --use-elo \
#           --dataset bioasq \
#           --output $OUTPUT_DIR/phase0/elo/bioasq/answer_variants.json \
#           --checkpoint-path $OUTPUT_DIR/phase0/elo/bioasq/checkpoint.json \
#           --num-variants 7 \
#           --batch-size 128

# Generate 5 variants for MEDIQA-AnS dataset:
#   SLURM:  sbatch $0 --dataset mediqaans --num-variants 5
#   PYTHON: python -m generation.phase0 --use-elo \
#           --dataset mediqaans \
#           --output $OUTPUT_DIR/phase0/elo/mediqaans/answer_variants.json \
#           --checkpoint-path $OUTPUT_DIR/phase0/elo/mediqaans/checkpoint.json \
#           --num-variants 5 \
#           --batch-size 128