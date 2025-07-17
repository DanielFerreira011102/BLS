#!/bin/bash
#SBATCH --job-name=ic_corpus_stats  # Name of the job visible in queue
#SBATCH --nodes=1                   # Number of compute nodes to allocate
#SBATCH --ntasks=1                  # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12          # CPU cores per task
#SBATCH --mem=40G                   # Memory allocation per node
#SBATCH --gres=gpu:1                # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu             # Compute partition/queue to use
#SBATCH --nodelist=gpu-srv-03       # Specific node to run on (can be overridden)
#SBATCH --chdir=/data/home/djbf/storage/bls/rq1  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq1/logs/%j/corpus_stats.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq1/logs/%j/corpus_stats.err   # Standard error file (%j = job ID)
#SBATCH --time=48:00:00             # Time limit (48 hours)

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

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

check_status() {
    if [ $? -ne 0 ]; then
        log_error "$1"
        exit 1
    fi
}

show_usage() {
    echo "Usage: sbatch $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --output-dir DIR       Directory to save statistics (default: corpus_stats)"
    echo "  --spacy-model MODEL    SpaCy model to use (default: en_core_web_trf)"
    echo "  --min-words NUM        Minimum words per text (default: 0)"
    echo "  --batch-size NUM       Batch size for processing (default: 1000)"
    echo "  --help                 Show this help message and exit"
    exit 1
}

#===============================================================================
# ARGUMENT PARSING
#===============================================================================

# Set default values
OUTPUT_DIR="corpus_stats2"
SPACY_MODEL="en_core_web_trf"
MIN_WORDS=0
BATCH_SIZE=1000

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --spacy-model)
            SPACY_MODEL="$2"
            shift 2
            ;;
        --min-words)
            MIN_WORDS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            log_error "Unknown argument: $1"
            show_usage
            ;;
    esac
done

#===============================================================================
# ENVIRONMENT SETUP
#===============================================================================

log_section "Environment Setup"

# Create logs directory for this job
LOGS_DIR="/data/home/djbf/storage/bls/rq1/logs/$SLURM_JOB_ID"
mkdir -p "$LOGS_DIR"
check_status "Failed to create logs directory"

log_info "Using the following parameters:"
log_info "Output directory: $OUTPUT_DIR"
log_info "SpaCy model: $SPACY_MODEL"
log_info "Minimum words: $MIN_WORDS"
log_info "Batch size: $BATCH_SIZE"

# Load required modules (adjust modules as needed)
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

# Activate virtual environment if necessary
log_info "Activating virtual environment..."
if [ -f "/data/home/djbf/bls/venv/bin/activate" ]; then
    source /data/home/djbf/bls/venv/bin/activate
    check_status "Failed to activate virtual environment"
fi

# Verify GPU availability
log_section "GPU Information"
nvidia-smi
check_status "Failed to query GPU information"

#===============================================================================
# EXECUTION
#===============================================================================

log_section "Starting corpus statistics computation"

# Define the Python script path
SCRIPT_PATH="/data/home/djbf/bls/rq1/metrics/utils/ic_corpus_stats.py"

# Check if the script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    log_error "Script not found: $SCRIPT_PATH"
    exit 1
fi

# Construct and run the Python command with arguments
PYTHON_CMD="python $SCRIPT_PATH"
PYTHON_CMD+=" --output-dir \"$OUTPUT_DIR\""
PYTHON_CMD+=" --spacy-model \"$SPACY_MODEL\""
PYTHON_CMD+=" --min-words $MIN_WORDS"
PYTHON_CMD+=" --batch-size $BATCH_SIZE"

log_info "Executing command: $PYTHON_CMD"
eval "$PYTHON_CMD"
execution_status=$?

if [ $execution_status -eq 0 ]; then
    log_info "Job completed successfully. Results saved to: $OUTPUT_DIR"
else
    log_error "Job failed with status code: $execution_status"
    exit $execution_status
fi

# Deactivate the virtual environment if activated
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

log_section "Job Completed"
log_info "Job completed at $(date)"