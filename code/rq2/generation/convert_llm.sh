#!/bin/bash
#SBATCH --job-name=bls_convert_llm  # Name of the job visible in queue
#SBATCH --nodes=1                   # Number of compute nodes to allocate
#SBATCH --ntasks=1                  # Number of tasks (processes) to create
#SBATCH --cpus-per-task=24          # CPU cores per task
#SBATCH --mem=180G                  # Memory allocation per node (increased for model loading)
#SBATCH --gres=gpu:1                # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu             # Compute partition/queue to use
#SBATCH --nodelist=gpu-srv-03       # Specific node to run on
#SBATCH --chdir=/data/home/djbf/storage/bls/rq2  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq2/logs/%j/model_conversion.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq2/logs/%j/model_conversion.err   # Standard error file (%j = job ID)
#SBATCH --time=48:00:00             # Time limit (48 hours)

#===============================================================================
# Biomedical Language Simplification (BLS) - LLM Model Conversion
# This script quantizes HuggingFace models and converts them to TurboMind format
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
    echo "Usage: sbatch $0 <model_name> <action>"
    echo ""
    echo "Arguments:"
    echo "  <model_name>  HuggingFace model name (e.g., aaditya/Llama3-OpenBioLLM-70B)"
    echo "  <action>      Operation to perform: quantize, convert, or both"
    echo ""
    echo "Actions:"
    echo "  quantize  Quantize the model using AWQ to INT4 precision"
    echo "  convert   Convert an already quantized model to TurboMind format"
    echo "  both      Perform both quantization and conversion in sequence"
    echo ""
    echo "Example:"
    echo "  sbatch $0 aaditya/Llama3-OpenBioLLM-70B both"
    exit 1
}

# Quantize model function
quantize_model() {
    local model_name=$1
    local output_dir=$2

    log_section "Starting Quantization"
    
    log_info "Model: $model_name"
    log_info "Output directory: $output_dir"
    log_info "Parameters: $WEIGHT_BITS-bit, group size $GROUP_SIZE, $CALIB_SAMPLES samples"
    
    lmdeploy lite auto_awq \
      "$model_name" \
      --calib-dataset "$CALIB_DATASET" \
      --calib-samples "$CALIB_SAMPLES" \
      --calib-seqlen "$CALIB_SEQLEN" \
      --w-bits "$WEIGHT_BITS" \
      --w-group-size "$GROUP_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --work-dir "$output_dir"
    
    check_status "Quantization failed"
    log_info "Quantization completed successfully"
    log_info "Quantized model saved to: $output_dir"
}

# Convert model function
convert_model() {
    local model_identifier=$1
    local quantized_dir=$2
    local output_dir=$3

    log_section "Starting Conversion to TurboMind Format"
    
    # Check if quantized model exists
    if [ ! -d "$quantized_dir" ]; then
        log_error "Quantized model directory $quantized_dir does not exist"
        log_error "Please ensure the quantized model is available before converting"
        exit 1
    fi
    
    log_info "Model identifier: $model_identifier"
    log_info "Input directory: $quantized_dir"
    log_info "Output directory: $output_dir"
    
    lmdeploy convert "$model_identifier" "$quantized_dir" \
      --dst-path "$output_dir" \
      --model-format awq \
      --group-size "$GROUP_SIZE"
    
    check_status "Conversion failed"
    log_info "Conversion completed successfully"
    log_info "TurboMind model saved to: $output_dir"
}

#===============================================================================
# DIRECTORY SETUP
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
BASE_DIR="$HOME_DIR/storage/bls"
MODELS_DIR="$BASE_DIR/resources/models"
LOGS_DIR="$BASE_DIR/logs/$SLURM_JOB_ID"

# Create required directories
mkdir -p "$LOGS_DIR"
mkdir -p "$MODELS_DIR"
check_status "Failed to create required directories"

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

# Check if two arguments are provided
if [ "$#" -ne 2 ]; then
    log_error "Invalid number of arguments"
    show_usage
fi

# Assign arguments
model_name=$1
action=$2

# Validate action
case "$action" in
    quantize|convert|both)
        # Valid action
        log_info "Action: $action"
        ;;
    *)
        log_error "Invalid action: $action"
        log_error "Action must be one of: quantize, convert, both"
        show_usage
        ;;
esac

log_info "Model: $model_name"

# Extract model identifier from model_name
identifier=$(basename "$model_name")
export HF_MODEL="$model_name"

#===============================================================================
# CONFIGURATION
#===============================================================================

log_section "Configuration"

# Quantization parameters
CALIB_DATASET="wikitext2"
CALIB_SAMPLES=256
CALIB_SEQLEN=1024
WEIGHT_BITS=4
GROUP_SIZE=128
BATCH_SIZE=1

# Set directories for quantized and TurboMind models
QUANTIZED_DIR="$MODELS_DIR/$identifier-AWQ-INT4"
TURBOMIND_DIR="$MODELS_DIR/$identifier-AWQ-INT4-TurboMind"

log_info "Calibration dataset: $CALIB_DATASET"
log_info "Calibration samples: $CALIB_SAMPLES"
log_info "Weight bits: $WEIGHT_BITS"
log_info "Group size: $GROUP_SIZE"
log_info "Quantized model directory: $QUANTIZED_DIR"
log_info "TurboMind model directory: $TURBOMIND_DIR"

#===============================================================================
# EXECUTION
#===============================================================================

log_section "Starting Job Execution"

log_info "Job ID: $SLURM_JOB_ID"
log_info "Running conversion with action: $action"

# Perform actions based on the specified action
case "$action" in
    quantize)
        quantize_model "$model_name" "$QUANTIZED_DIR"
        ;;
    convert)
        convert_model "$identifier" "$QUANTIZED_DIR" "$TURBOMIND_DIR"
        ;;
    both)
        quantize_model "$model_name" "$QUANTIZED_DIR"
        convert_model "$identifier" "$QUANTIZED_DIR" "$TURBOMIND_DIR"
        ;;
esac

#===============================================================================
# COMPLETION
#===============================================================================

log_section "Job Completed Successfully"

if [[ "$action" == "quantize" || "$action" == "both" ]]; then
    log_info "Quantized model saved to: $QUANTIZED_DIR"
fi

if [[ "$action" == "convert" || "$action" == "both" ]]; then
    log_info "TurboMind model saved to: $TURBOMIND_DIR"
fi

log_info "Job ID: $SLURM_JOB_ID completed at $(date)"