#!/bin/bash
#SBATCH --job-name=llmify_batch   # Name of the job visible in queue
#SBATCH --nodes=1                 # Number of compute nodes to allocate
#SBATCH --ntasks=1                # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12        # CPU cores per task
#SBATCH --mem=40G                 # Memory allocation per node - 60GB
#SBATCH --gres=gpu:1              # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu           # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq1  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq1/logs/batch_%j.out  # Standard output file
#SBATCH --error=/data/home/djbf/storage/bls/rq1/logs/batch_%j.err   # Standard error file
#SBATCH --time=72:00:00           # Maximum run time (72 hours, adjust as needed)

#===============================================================================
# LLMify Batch Runner - RECOVERY RUN - Only missing configurations
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
        return 1
    fi
    return 0
}

#===============================================================================
# ENVIRONMENT SETUP
#===============================================================================

log_section "Environment Setup"

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq2"
BASE_DIR="$HOME_DIR/storage/bls"
WORK_DIR="$BASE_DIR/rq2"
LOGS_DIR="$WORK_DIR/logs/$SLURM_JOB_ID"
OUTPUT_BASE_DIR="$WORK_DIR/outputs/llmify"

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

# Activate virtual environment
log_info "Activating virtual environment..."
source "$HOME_DIR/bls/venv/bin/activate"
check_status "Failed to activate virtual environment"

# Verify GPU availability
log_section "GPU Information"
nvidia-smi
check_status "Failed to query GPU information"

#===============================================================================
# CONFIGURATION RUNNER
#===============================================================================

# Function to run a specific configuration
run_configuration() {
    local config_name="$1"
    local script="$2"
    local model_type="$3"
    local mse_weight="$4"
    local ce_weight="$5"
    local kl_weight="$6"
    local soft_ce_weight="$7"
    local attention_heads="${8:-2}"     # Default to 2 if not provided
    local dropout_rate="${9:-0.2}"      # Default to 0.2 if not provided
    local config_log_dir="$LOGS_DIR/$config_name"

    log_section "Running Configuration: $config_name"

    # Create log directory for this configuration
    mkdir -p "$config_log_dir"

    # Construct full script path
    local script_path="$SOURCE_DIR/llmify/$script"
    if [ ! -f "$script_path" ]; then
        log_error "Script not found: $script_path"
        return 1
    fi

    # Set fixed parameters
    local fixed_args="--model-name \"$MODEL_NAME\" --model-type $model_type --file-patterns \"$FILE_PATTERNS\" --batch-size $BATCH_SIZE --max-length $MAX_LENGTH --learning-rate $LEARNING_RATE --epochs $EPOCHS --attention-heads $attention_heads --dropout-rate $dropout_rate"

    # Construct output directory based on script name
    local loss_suffix="mse_${mse_weight}+kl_${kl_weight}+ce_${ce_weight}+soft_${soft_ce_weight}+h_${attention_heads}+d_${dropout_rate}"
    local out_dir=""
    
    if [[ "$script" == *"ordinal"* ]]; then
        out_dir="$OUTPUT_BASE_DIR/ordinal/$model_type/${loss_suffix}"
    else
        out_dir="$OUTPUT_BASE_DIR/default/$model_type/${loss_suffix}"
    fi
    
    mkdir -p "$out_dir"

    # Construct the full command
    local cmd="python $script_path $fixed_args --mse-weight $mse_weight --ce-weight $ce_weight --kl-weight $kl_weight --soft-ce-weight $soft_ce_weight --output-dir \"$out_dir\""

    # Log the command
    log_info "Executing: $cmd"

    # Run the command and capture stdout/stderr to log files
    eval "$cmd" > "$config_log_dir/stdout.log" 2> "$config_log_dir/stderr.log"
    local status=$?

    if [ $status -eq 0 ]; then
        log_info "Configuration $config_name completed successfully"
    else
        log_error "Configuration $config_name failed with exit code $status"
        log_error "Check logs at: $config_log_dir/stderr.log"
    fi

    return $status
}

log_section "Starting Recovery Batch Execution - Job ID: $SLURM_JOB_ID"

# Set all fixed parameters
MODEL_NAME="kamalkraj/BioSimCSE-BioLinkBERT-BASE"
FILE_PATTERNS="/data/home/djbf/storage/bls/rq1/outputs/phase0/*/llm/5/*/readability_metrics.json"
BATCH_SIZE=32
MAX_LENGTH=512
LEARNING_RATE=2e-5
EPOCHS=15

#-------------------------------------------------------------------------------
# Run Set 1: Regular model with different loss functions
#-------------------------------------------------------------------------------
log_section "Configuration Set 1: Standard Model with Different Loss Functions"

# Standard model with MSE loss
# run_configuration "standard_mse" "llmify.py" "standard" 1.0 0.0 0.0 0.0 2 0.2

# Standard model with CE loss
# run_configuration "standard_ce" "llmify.py" "standard" 0.0 1.0 0.0 0.0 2 0.2

# Standard model with KL loss
# run_configuration "standard_kl" "llmify.py" "standard" 0.0 0.0 1.0 0.0 2 0.2

# Standard model with soft CE loss
# run_configuration "standard_soft_ce" "llmify.py" "standard" 0.0 0.0 0.0 1.0 2 0.2

# Standard model with KL+MSE loss
# run_configuration "standard_kl_mse" "llmify.py" "standard" 1.0 0.0 1.0 0.0 2 0.2

# Standard model with soft CE + MSE loss
# run_configuration "standard_soft_ce_mse" "llmify.py" "standard" 1.0 0.0 0.0 1.0 2 0.2

# Standard model with combined losses
# run_configuration "standard_combined" "llmify.py" "standard" 1.0 1.0 1.0 1.0 2 0.2

#-------------------------------------------------------------------------------
# Run Set 2: Improved model with different loss functions
#-------------------------------------------------------------------------------
log_section "Configuration Set 2: Improved Model with Different Loss Functions"

# Improved model with MSE loss
# run_configuration "improved_mse" "llmify.py" "improved" 1.0 0.0 0.0 0.0 2 0.2

# Improved model with CE loss
# run_configuration "improved_ce" "llmify.py" "improved" 0.0 1.0 0.0 0.0 2 0.2

# Improved model with KL loss
# run_configuration "improved_kl" "llmify.py" "improved" 0.0 0.0 1.0 0.0 2 0.2

# Improved model with soft CE loss
# run_configuration "improved_soft_ce" "llmify.py" "improved" 0.0 0.0 0.0 1.0 2 0.2

# Improved model with KL + MSE loss
# run_configuration "improved_kl_mse" "llmify.py" "improved" 1.0 0.0 1.0 0.0 2 0.2

# Improved model with soft CE + MSE loss
# run_configuration "improved_soft_ce_mse" "llmify.py" "improved" 1.0 0.0 0.0 1.0 2 0.2

# Improved model with combined losses
# run_configuration "improved_combined" "llmify.py" "improved" 1.0 1.0 1.0 1.0 2 0.2

#-------------------------------------------------------------------------------
# Run Set 3: Ordinal regression with standard model
#-------------------------------------------------------------------------------
log_section "Configuration Set 3: Ordinal Regression with Standard Model"

# Ordinal with MSE loss
# run_configuration "ordinal_standard_mse" "llmify_ordinal.py" "standard" 1.0 0.0 0.0 0.0 2 0.2

# Ordinal with CE loss
# run_configuration "ordinal_standard_ce" "llmify_ordinal.py" "standard" 0.0 1.0 0.0 0.0 2 0.2

# Ordinal with KL loss
# run_configuration "ordinal_standard_kl" "llmify_ordinal.py" "standard" 0.0 0.0 1.0 0.0 2 0.2

# Ordinal with soft CE loss
# run_configuration "ordinal_standard_soft_ce" "llmify_ordinal.py" "standard" 0.0 0.0 0.0 1.0 2 0.2

# Ordinal with KL + MSE loss
# run_configuration "ordinal_standard_kl_mse" "llmify_ordinal.py" "standard" 1.0 0.0 1.0 0.0 2 0.2

# Ordinal with soft CE + MSE loss
# run_configuration "ordinal_standard_soft_ce_mse" "llmify_ordinal.py" "standard" 1.0 0.0 0.0 1.0 2 0.2

# Ordinal with combined losses
# run_configuration "ordinal_standard_combined" "llmify_ordinal.py" "standard" 1.0 1.0 1.0 1.0 2 0.2

#-------------------------------------------------------------------------------
# Run Set 4: Ordinal regression with improved model
#-------------------------------------------------------------------------------
log_section "Configuration Set 4: Ordinal Regression with Improved Model"

# Ordinal with MSE loss
# run_configuration "ordinal_improved_mse" "llmify_ordinal.py" "improved" 1.0 0.0 0.0 0.0 2 0.2

# Ordinal with CE loss
# run_configuration "ordinal_improved_ce" "llmify_ordinal.py" "improved" 0.0 1.0 0.0 0.0 2 0.2

# Ordinal with KL loss
# run_configuration "ordinal_improved_kl" "llmify_ordinal.py" "improved" 0.0 0.0 1.0 0.0 2 0.2

# Ordinal with soft CE loss
# run_configuration "ordinal_improved_soft_ce" "llmify_ordinal.py" "improved" 0.0 0.0 0.0 1.0 2 0.2

# Ordinal with KL + MSE loss
# run_configuration "ordinal_improved_kl_mse" "llmify_ordinal.py" "improved" 1.0 0.0 1.0 0.0 2 0.2

# Ordinal with soft CE + MSE loss
# run_configuration "ordinal_improved_soft_ce_mse" "llmify_ordinal.py" "improved" 1.0 0.0 0.0 1.0 2 0.2

# Ordinal with combined losses
# run_configuration "ordinal_improved_combined" "llmify_ordinal.py" "improved" 1.0 1.0 1.0 1.0 2 0.2

#-------------------------------------------------------------------------------
# Run Set 5: Alternate architecture parameters (attention heads and dropout)
#-------------------------------------------------------------------------------
log_section "Configuration Set 5: Varying Attention Heads and Dropout Rate"

# Default/improved model with MSE+KL loss - 1 attention head
# run_configuration "improved_mse_kl_head1" "llmify.py" "improved" 1.0 0.0 1.0 0.0 1 0.2

# Default/improved model with MSE+KL loss - 4 attention heads
# run_configuration "improved_mse_kl_head4" "llmify.py" "improved" 1.0 0.0 1.0 0.0 4 0.2

# Ordinal/improved model with MSE loss - 1 attention head
# run_configuration "ordinal_improved_mse_head1" "llmify_ordinal.py" "improved" 1.0 0.0 0.0 0.0 1 0.2

# Ordinal/improved model with MSE loss - 4 attention heads
run_configuration "ordinal_improved_mse_head4" "llmify_ordinal.py" "improved" 1.0 0.0 0.0 0.0 4 0.2


#===============================================================================
# COMPLETION
#===============================================================================
log_section "Batch Job Completed"
log_info "All configurations have been processed."
log_info "Job ID: $SLURM_JOB_ID completed at $(date)"

# Deactivate the virtual environment
deactivate

exit 0