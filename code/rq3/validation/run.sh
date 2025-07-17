#!/bin/bash
#SBATCH --job-name=claude_eval       # Name of the job visible in queue
#SBATCH --nodes=1                    # Number of compute nodes to allocate
#SBATCH --ntasks=1                   # Number of tasks (processes) to create
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=16G                    # Memory allocation per node
#SBATCH --partition=cpu              # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq3  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq3/logs/%j/claude_eval.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq3/logs/%j/claude_eval.err   # Standard error file (%j = job ID)
#SBATCH --time=8:00:00               # Time limit (8 hours)

#===============================================================================
# Biomedical Language Simplification (BLS) - Claude Response Evaluation
# This script evaluates response quality using the Claude API
#===============================================================================

# Global variables
INPUT_FILE=""
OUTPUT_FILE=""
CLAUDE_API_KEY="your_claude_api_key_here"  # Replace with your actual API key
BATCH_SIZE=32

# Directory and file paths
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq3"
BASE_DIR="$HOME_DIR/storage/bls"
INPUT_BASE_DIR="$BASE_DIR/rq3/outputs/phase4"
OUTPUT_BASE_DIR="$BASE_DIR/rq3/outputs/phase5"
LOGS_DIR=""
EVAL_SCRIPT="$SOURCE_DIR/validation/phase5.py"

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
    return 0
}

# Show usage information
show_usage() {
    echo "Usage: sbatch run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input-file FILE         Path to the input CSV file containing questions and responses"
    echo "  --output-file FILE        Path to save the output CSV file with evaluation results"
    echo "  --api-key KEY             Claude API key (if not provided, will use a default example key)"
    echo "  --batch-size SIZE         Batch size for API requests (default: 32)"
    echo ""
    echo "Examples:"
    echo "  sbatch run.sh --input-file /path/to/input.csv --output-file /path/to/output.csv"
    echo "  sbatch run.sh --input-file /path/to/input.csv --output-file /path/to/output.csv --batch-size 64"
    exit 1
}

#===============================================================================
# SETUP FUNCTIONS
#===============================================================================

# Parse command line arguments
parse_arguments() {
    log_section "Parsing Arguments"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --input-file)
                INPUT_FILE="$2"
                shift 2
                ;;
            --output-file)
                OUTPUT_FILE="$2"
                shift 2
                ;;
            --api-key)
                CLAUDE_API_KEY="$2"
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
                log_error "Unknown option: $1"
                show_usage
                ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$INPUT_FILE" ]; then
        log_error "Required argument: --input-file"
        show_usage
    fi
    
    if [ -z "$OUTPUT_FILE" ]; then
        log_error "Required argument: --output-file"
        show_usage
    fi
    
    log_info "Input file: $INPUT_FILE"
    log_info "Output file: $OUTPUT_FILE"
    log_info "API key: ${CLAUDE_API_KEY:0:8}..." # Only show beginning of API key for security
    log_info "Batch size: $BATCH_SIZE"
}

# Setup environment and directories
setup_environment() {
    log_section "Environment Setup"
    
    # Set up job-specific directories
    LOGS_DIR="$BASE_DIR/rq3/logs/$SLURM_JOB_ID"
    
    # Create required directories
    mkdir -p "$OUTPUT_BASE_DIR" "$LOGS_DIR"
    check_status "Failed to create base directories"
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
    mkdir -p "$OUTPUT_DIR"
    check_status "Failed to create output directory: $OUTPUT_DIR"
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        log_error "Input file not found: $INPUT_FILE"
        exit 1
    fi
    
    # Check if evaluation script exists
    if [ ! -f "$EVAL_SCRIPT" ]; then
        log_error "Evaluation script not found: $EVAL_SCRIPT"
        exit 1
    fi
    
    # Load required system modules
    log_info "Loading required modules..."
    module load python
    check_status "Failed to load Python module"
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    check_status "Failed to activate virtual environment"
    
    # Set environment variables
    export ANTHROPIC_API_KEY="$CLAUDE_API_KEY"
    
    log_info "Environment setup complete"
}

#===============================================================================
# EVALUATION FUNCTIONS
#===============================================================================

# Run the Claude evaluation
run_evaluation() {
    log_section "Running Claude Evaluation"
    
    start_time=$(date +%s)
    
    # Build the command
    cmd="python $EVAL_SCRIPT \
        --input-file \"$INPUT_FILE\" \
        --output-file \"$OUTPUT_FILE\" \
        --api-key \"$CLAUDE_API_KEY\" \
        --batch-size $BATCH_SIZE"
    
    # Log and execute the command
    log_info "Executing: $cmd"
    eval "$cmd"
    check_status "Claude evaluation failed"
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    hours=$((elapsed_time / 3600))
    minutes=$(( (elapsed_time % 3600) / 60 ))
    seconds=$((elapsed_time % 60))
    
    log_info "Evaluation completed in ${hours}h ${minutes}m ${seconds}s"
    
    return 0
}

# Report evaluation results
report_results() {
    log_section "Evaluation Results"
    
    # Check if output file was created
    if [ -f "$OUTPUT_FILE" ]; then
        # Count the number of evaluated samples (excluding header)
        sample_count=$(($(wc -l < "$OUTPUT_FILE") - 1))
        log_info "Evaluated $sample_count samples"

        # Check for dimensions.json and display
        dimensions_file="$(dirname "$OUTPUT_FILE")/dimensions.json"
        if [ -f "$dimensions_file" ]; then
            log_info "Dimensions file created: $dimensions_file"
            log_info "Evaluation dimensions:"
            grep -v "scale" "$dimensions_file" | sed 's/[{}"]//g' | sed 's/,//g' | sed 's/dimensions://g'
        else
            log_warn "Dimensions file not found: $dimensions_file"
        fi

        # Calculate average scores
        log_info "Calculating average scores..."
        python - <<EOF
import pandas as pd

df = pd.read_csv("$OUTPUT_FILE")
score_cols = [col for col in df.columns if col.startswith("score_")]
if score_cols:
    print("Average scores:")
    for col in score_cols:
        name = col.replace("score_", "").title()
        avg = df[col].mean()
        print(f"  {name}: {avg:.2f}")
EOF

    else
        log_warn "Output file not found: $OUTPUT_FILE"
    fi
}


#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Setup environment
    setup_environment
    
    # Start evaluation
    log_section "Starting Claude Evaluation - Job ID: $SLURM_JOB_ID"
    
    # Run evaluation
    run_evaluation
    
    # Report results
    report_results
    
    # Completion
    log_section "Job Completed Successfully"
    
    log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
    log_info "Results saved to: $OUTPUT_FILE"
    
    # Deactivate the virtual environment
    deactivate
}

# Execute main with all arguments
main "$@"

exit 0

#===============================================================================
# USAGE EXAMPLES (For reference only, not executed)
#===============================================================================

# Run evaluation with default batch size:
#   SLURM:  sbatch run.sh --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline/metrics_df.csv" \
#                               --output-file "/data/home/djbf/storage/bls/rq3/outputs/phase5/baseline/evaluation_results.csv"
#   PYTHON: python "/data/home/djbf/bls/rq3/validation/phase5.py" \
#           --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline/metrics_df.csv" \
#           --output-file "/data/home/djbf/storage/bls/rq3/outputs/phase5/baseline/evaluation_results.csv" \
#           --api-key "sk-ant-api03-examplekey12345abcdefghijklmnopqrstuvwxyz0123456789" \
#           --batch-size 32

# Run evaluation with custom batch size:
#   SLURM:  sbatch run.sh --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline/metrics_df.csv" \
#                               --output-file "/data/home/djbf/storage/bls/rq3/outputs/phase5/baseline/evaluation_results.csv" \
#                               --batch-size 64
#   PYTHON: python "/data/home/djbf/bls/rq3/validation/phase5.py" \
#           --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline/metrics_df.csv" \
#           --output-file "/data/home/djbf/storage/bls/rq3/outputs/phase5/baseline/evaluation_results.csv" \
#           --api-key "sk-ant-api03-examplekey12345abcdefghijklmnopqrstuvwxyz0123456789" \
#           --batch-size 64