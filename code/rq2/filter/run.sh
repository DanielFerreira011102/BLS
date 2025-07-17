#!/bin/bash
#SBATCH --job-name=bls_phase_runner  # Name of the job visible in queue
#SBATCH --nodes=1                 # Number of compute nodes to allocate
#SBATCH --ntasks=1                # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12        # CPU cores per task
#SBATCH --mem=40G                 # Memory allocation per node - 40GB
#SBATCH --gres=gpu:1              # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu           # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq2  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq2/logs/%j/logs.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq2/logs/%j/logs.err   # Standard error file (%j = job ID)
#SBATCH --time=72:00:00           # Maximum run time (72 hours)

#===============================================================================
# BLS Phase Runner - Run Phase 2 or Phase 4
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

# Verify GPU availability
log_section "GPU Information"
nvidia-smi
check_status "Failed to query GPU information"

#===============================================================================
# Argument Parsing
#===============================================================================

log_section "Parsing Arguments"

# Check if phase argument is provided
if [ $# -ne 1 ]; then
    log_error "Exactly one argument required: phase (2 or 4)"
    log_error "Usage: sbatch $0 <phase>"
    exit 1
fi

PHASE="$1"

# Validate phase argument
if [ "$PHASE" != "2" ] && [ "$PHASE" != "4" ]; then
    log_error "Invalid phase: $PHASE. Must be '2' or '4'"
    exit 1
fi

log_info "Selected phase: $PHASE"

#===============================================================================
# Phase Execution
#===============================================================================

log_section "Starting Phase $PHASE Execution - Job ID: $SLURM_JOB_ID"

# Define script path and command based on phase
if [ "$PHASE" = "2" ]; then
    SCRIPT_PATH="$SOURCE_DIR/filter/phase2.py"
    INPUT_FILE="$WORK_DIR/outputs/phase1/liveqa+medicationqa+mediqaans+bioasq+medquad/aggregated_metrics.json"
    OUTPUT_DIR="$WORK_DIR/outputs/phase2"
    CMD="python $SCRIPT_PATH \
        --input-file $INPUT_FILE \
        --output-dir $OUTPUT_DIR"
elif [ "$PHASE" = "4" ]; then
    SCRIPT_PATH="$SOURCE_DIR/filter/phase4.py"
    METRICS_FILE="$WORK_DIR/outputs/phase2/metrics_df.csv"
    SCORES_FILE="$WORK_DIR/outputs/phase3/merged/scores_df.csv"
    OUTPUT_DIR="$WORK_DIR/outputs/phase4"
    RULES_FILE="$SOURCE_DIR/filter/rules/rules3.json"
    CMD="python $SCRIPT_PATH \
        --metrics-file $METRICS_FILE \
        --scores-file $SCORES_FILE \
        --output-dir $OUTPUT_DIR \
        --rules-file $RULES_FILE"
fi

# Verify script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    log_error "Script not found: $SCRIPT_PATH"
    exit 1
fi

# Verify input files
if [ "$PHASE" = "2" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        log_error "Input file not found: $INPUT_FILE"
        exit 1
    fi
elif [ "$PHASE" = "4" ]; then
    if [ ! -f "$METRICS_FILE" ]; then
        log_error "Metrics file not found: $METRICS_FILE"
        exit 1
    fi
    if [ ! -f "$SCORES_FILE" ]; then
        log_error "Scores file not found: $SCORES_FILE"
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
check_status "Failed to create output directory: $OUTPUT_DIR"

# Log the command
log_info "Executing: $CMD"

# Run the command
eval "$CMD"
status=$?

if [ $status -eq 0 ]; then
    log_info "Phase $PHASE execution completed successfully"
else
    log_error "Phase $PHASE execution failed with exit code $status"
    log_error "Check logs at: $LOGS_DIR/logs.err"
    exit $status
fi

#===============================================================================
# Completion
#===============================================================================

log_section "Batch Job Completed"
log_info "Phase $PHASE execution processed."
log_info "Job ID: $SLURM_JOB_ID completed at $(date)"

# Deactivate virtual environment
deactivate

exit 0