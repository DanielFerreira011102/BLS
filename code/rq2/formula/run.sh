#!/bin/bash
#SBATCH --job-name=bls_phase5     # Name of the job visible in queue
#SBATCH --nodes=1                 # Number of compute nodes to allocate
#SBATCH --ntasks=1                # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12        # CPU cores per task
#SBATCH --mem=40G                 # Memory allocation per node - 40GB
#SBATCH --partition=cpu           # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq2  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq2/logs/%j/logs.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq2/logs/%j/logs.err   # Standard error file (%j = job ID)
#SBATCH --time=72:00:00           # Maximum run time (72 hours)

#===============================================================================
# BLS Phase 5 Runner - Complexity Score Generation
#===============================================================================

#===============================================================================
# Helper Functions
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
        return 1
    fi
    return 0
}

#===============================================================================
# Environment Setup
#===============================================================================

log_section "Environment Setup"

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq2"
BASE_DIR="$HOME_DIR/storage/bls"
RESOURCE_DIR="$BASE_DIR/resources"
WORK_DIR="$BASE_DIR/rq2"
LOGS_DIR="$WORK_DIR/logs/$SLURM_JOB_ID"
NLP_DIR="$RESOURCE_DIR/nlp"

# Create logs directory for this job
mkdir -p "$LOGS_DIR"
check_status "Failed to create logs directory"

# Load required system modules
log_info "Loading required modules..."
module load python
check_status "Failed to load Python module"

module load gcc/13.2.0-l6taobu
check_status "Failed to load GCC module"

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

#===============================================================================
# Argument Parsing
#===============================================================================

log_section "Parsing Arguments"

# Default values
DEFAULT_MODEL_DIR="$BASE_DIR/rq1/outputs/phase4/combined_external_eval/stable16/experiments/penalty_elasticnet/models"
MODEL_DIR="$DEFAULT_MODEL_DIR"

# Check if a custom model directory is provided
if [ $# -ge 1 ]; then
    MODEL_DIR="$1"
    log_info "Using custom model directory: $MODEL_DIR"
else
    log_info "Using default model directory: $MODEL_DIR"
fi

# Verify model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    log_error "Model directory not found: $MODEL_DIR"
    exit 1
fi

# Verify model directory contains required files
if [ ! -f "$MODEL_DIR/complexity_pipeline.joblib" ]; then
    log_error "Model pipeline file not found: $MODEL_DIR/complexity_pipeline.joblib"
    exit 1
fi

if [ ! -f "$MODEL_DIR/model_features.json" ]; then
    log_error "Model features file not found: $MODEL_DIR/model_features.json"
    exit 1
fi

#===============================================================================
# Phase 5 Execution
#===============================================================================

log_section "Starting Phase 5 Execution - Job ID: $SLURM_JOB_ID"

# Define script path and command
SCRIPT_PATH="$SOURCE_DIR/formula/phase5.py"
INPUT_FILE="$WORK_DIR/outputs/phase4/filtered_df.csv"
OUTPUT_DIR="$WORK_DIR/outputs/phase5"

# Verify script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    log_error "Script not found: $SCRIPT_PATH"
    exit 1
fi

# Verify input file
if [ ! -f "$INPUT_FILE" ]; then
    log_error "Input file not found: $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
check_status "Failed to create output directory: $OUTPUT_DIR"

# Construct and log the command
CMD="python $SCRIPT_PATH --input-file $INPUT_FILE --model-dir $MODEL_DIR --output-dir $OUTPUT_DIR"
log_info "Executing: $CMD"

# Run the command
eval "$CMD"
status=$?

if [ $status -eq 0 ]; then
    log_info "Phase 5 execution completed successfully"
else
    log_error "Phase 5 execution failed with exit code $status"
    log_error "Check logs at: $LOGS_DIR/logs.err"
    exit $status
fi

#===============================================================================
# Completion
#===============================================================================

log_section "Batch Job Completed"
log_info "Phase 5 execution processed."
log_info "Job ID: $SLURM_JOB_ID completed at $(date)"

# Deactivate virtual environment
deactivate

exit 0