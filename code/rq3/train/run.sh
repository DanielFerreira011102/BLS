#!/bin/bash
#SBATCH --job-name=complexity_train   # Name of the job visible in queue
#SBATCH --nodes=1                     # Number of compute nodes to allocate
#SBATCH --ntasks=1                    # Number of tasks (processes) to create
#SBATCH --cpus-per-task=16            # CPU cores per task
#SBATCH --mem=90G                     # Memory allocation per node - 40GB
#SBATCH --gres=gpu:1                  # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu               # Compute partition/queue to use
#SBATCH --nodelist=gpu-srv-03         # Specific node to run on (optional)
#SBATCH --chdir=/data/home/djbf/storage/bls/rq3  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq3/logs/%j/logs.out  # Standard output file
#SBATCH --error=/data/home/djbf/storage/bls/rq3/logs/%j/logs.err   # Standard error file
#SBATCH --time=72:00:00               # Maximum run time (72 hours, adjust as needed)

#===============================================================================
# Complexity Training - RQ3
#===============================================================================

# Global variables
TEST_MODE=${TEST_MODE:-false}  # Default to false if not set

# Directory and file paths
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq3/train"
BASE_DIR="$HOME_DIR/storage/bls"
WORK_DIR="$BASE_DIR/rq3"
LOGS_DIR=""
DATA_PATH="/data/home/djbf/storage/bls/rq2/outputs/phase5/scored_df.csv"
OUTPUT_DIR=""
SCRIPT_PATH=""

# API keys and tokens
WANDB_KEY="your_wandb_api_key_here"
HF_TOKEN="your_huggingface_token_here"

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
# SETUP FUNCTIONS
#===============================================================================

# Setup environment and directories
setup_environment() {
    log_section "Environment Setup"
    
    # Set up paths
    SCRIPT_PATH="$SOURCE_DIR/phase0.py"
    LOGS_DIR="$WORK_DIR/logs/$SLURM_JOB_ID"
    OUTPUT_DIR="$WORK_DIR/outputs/models/complexity_$(date +%Y%m%d_%H%M%S)"
    
    # Create necessary directories
    mkdir -p "$LOGS_DIR"
    check_status "Failed to create logs directory"
    
    mkdir -p "$OUTPUT_DIR"
    check_status "Failed to create output directory"
    
    # Verify script exists
    if [ ! -f "$SCRIPT_PATH" ]; then
        log_error "Script not found: $SCRIPT_PATH"
        exit 1
    fi
    
    # Verify data file exists
    if [ ! -f "$DATA_PATH" ]; then
        log_error "Data file not found: $DATA_PATH"
        exit 1
    fi
    
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
    
    # Set API keys
    export WANDB_API_KEY="$WANDB_KEY"
    export HUGGINGFACE_TOKEN="$HF_TOKEN"
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    check_status "Failed to activate virtual environment"
    
    log_info "Environment setup complete"
}

# Verify GPU availability
check_gpu() {
    log_section "GPU Information"
    
    nvidia-smi
    check_status "Failed to query GPU information"
    
    # Extract GPU information
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    
    log_info "Using GPU: $gpu_name with $gpu_memory"
    return 0
}

#===============================================================================
# TRAINING FUNCTIONS
#===============================================================================

# Build the training command
build_training_command() {
    local cmd=""
    
    if [ "$TEST_MODE" = "true" ]; then
        log_warn "Running in TEST MODE with reduced dataset"
        cmd="python $SCRIPT_PATH \
          --data-path $DATA_PATH \
          --output-dir $OUTPUT_DIR \
          --model-name unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
          --batch-size 2 \
          --gradient-accumulation-steps 4 \
          --save-steps 500 \
          --eval-steps 500 \
          --logging-steps 100 \
          --qualitative-test-steps 50"
    else
        # Full training run
        cmd="python $SCRIPT_PATH \
          --data-path $DATA_PATH \
          --output-dir $OUTPUT_DIR \
          --model-name unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    fi
    
    echo "$cmd"
}

# Run the training process
run_training() {
    log_section "Starting Complexity Training - Job ID: $SLURM_JOB_ID"
    
    # Build command
    local cmd=$(build_training_command)
    
    # Log the command
    log_info "Executing: $cmd"
    
    # Run the command
    start_time=$(date +%s)
    
    eval "$cmd"
    status=$?
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    hours=$((elapsed_time / 3600))
    minutes=$(( (elapsed_time % 3600) / 60 ))
    seconds=$((elapsed_time % 60))
    
    # Check status
    if [ $status -eq 0 ]; then
        log_info "Complexity training completed successfully"
        log_info "Training time: ${hours}h ${minutes}m ${seconds}s"
        return 0
    else
        log_error "Complexity training failed with exit code $status"
        return $status
    fi
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    # Setup environment
    setup_environment
    
    # Check GPU availability
    check_gpu
    
    # Run training
    run_training
    status=$?
    
    # Completion
    if [ $status -eq 0 ]; then
        log_section "Batch Job Completed"
        log_info "Complexity training processed successfully."
        log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
        log_info "Output saved to: $OUTPUT_DIR"
    else
        log_section "Batch Job Failed"
        log_error "Complexity training failed with exit code $status"
        log_info "Job ID: $SLURM_JOB_ID failed at $(date)"
    fi
    
    # Deactivate the virtual environment
    deactivate
    
    return $status
}

# Execute main function
main

exit $?