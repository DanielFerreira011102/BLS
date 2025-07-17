#!/bin/bash
#SBATCH --job-name=phase4_metrics     # Name of the job visible in queue
#SBATCH --nodes=1                     # Number of compute nodes to allocate
#SBATCH --ntasks=1                    # Number of tasks (processes) to create
#SBATCH --cpus-per-task=4             # CPU cores per task
#SBATCH --mem=16G                     # Memory allocation per node
#SBATCH --partition=cpu               # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq3  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq3/logs/%j/phase4_metrics.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq3/logs/%j/phase4_metrics.err   # Standard error file (%j = job ID)
#SBATCH --time=4:00:00                # Time limit (4 hours)

#===============================================================================
# Biomedical Language Simplification (BLS) - Phase 4 Metrics Analysis
# This script analyzes readability metrics from CSV files
#===============================================================================

# Global variables
DATASETS=""
OUTPUT_DIR=""
process_datasets=()

# Directory and file paths
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq3"
BASE_DIR="$HOME_DIR/storage/bls"
INPUT_BASE_DIR="$BASE_DIR/rq3/outputs/phase3"
OUTPUT_BASE_DIR="$BASE_DIR/rq3/outputs/phase4"
LOGS_DIR=""
THRESHOLDS_FILE="$SOURCE_DIR/formula/thresholds/quantile.json"
MODEL_DIR="$BASE_DIR/rq1/outputs/phase4/combined_external_eval/stable16/experiments/penalty_elasticnet/models"

# Define available datasets
AVAILABLE_DATASETS=("baseline" "fewshot" "finetuned" "finetunedIt" "claude" "baseline+fewshot+finetuned+finetunedIt+claude" "all")

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
    echo "  --datasets DATASETS      Datasets to analyze, comma-separated"
    echo "                           (baseline,fewshot,finetuned,finetunedIt,claude,"
    echo "                           baseline+fewshot+finetuned+finetunedIt+claude)"  
    echo "                           or 'all' for all datasets"
    echo "  --output-dir DIR         Base directory for output files"
    echo "                           (default: /data/home/djbf/storage/bls/rq3/outputs/phase4)"
    echo ""
    echo "Examples:"
    echo "  sbatch run.sh --datasets all"
    echo "  sbatch run.sh --datasets baseline,fewshot"
    echo "  sbatch run.sh --datasets baseline+fewshot+finetuned+finetunedIt+claude"
    exit 1
}

#===============================================================================
# SETUP FUNCTIONS
#===============================================================================

# Parse command line arguments
parse_arguments() {
    log_section "Argument Parsing"
    
    # Default values
    OUTPUT_DIR="$OUTPUT_BASE_DIR"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --datasets) DATASETS="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --help|-h) show_usage ;;
            *) log_error "Unknown argument: $1"; show_usage ;;
        esac
    done
    
    # Validate required arguments
    if [ -z "$DATASETS" ]; then
        log_error "Required argument: --datasets"
        show_usage
    fi
    
    # Determine which datasets to process
    if [ "$DATASETS" == "all" ]; then
        process_datasets=("${AVAILABLE_DATASETS[@]}")
    else
        IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
        for dataset in "${DATASET_LIST[@]}"; do
            # Check if dataset is valid
            valid_dataset=false
            for available_dataset in "${AVAILABLE_DATASETS[@]}"; do
                if [ "$dataset" == "$available_dataset" ]; then
                    valid_dataset=true
                    process_datasets+=("$dataset")
                    break
                fi
            done
            
            if [ "$valid_dataset" != true ]; then
                log_warn "Dataset '$dataset' is not recognized. Valid datasets include: ${AVAILABLE_DATASETS[*]}"
            fi
        done
        
        if [ ${#process_datasets[@]} -eq 0 ]; then
            log_error "No valid datasets specified"
            exit 1
        fi
    fi
    
    log_info "Datasets to process: ${process_datasets[*]}"
    log_info "Output directory: $OUTPUT_DIR"
}

# Setup environment and directories
setup_environment() {
    log_section "Environment Setup"
    
    # Set up job-specific directories
    LOGS_DIR="$BASE_DIR/rq3/logs/$SLURM_JOB_ID"
    
    # Create required directories
    mkdir -p "$OUTPUT_BASE_DIR" "$LOGS_DIR"
    check_status "Failed to create output directories"
    
    # Load required system modules
    log_info "Loading required modules..."
    module load python
    check_status "Failed to load Python module"
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    check_status "Failed to activate virtual environment"
    
    # Check if thresholds file exists
    if [ ! -f "$THRESHOLDS_FILE" ]; then
        log_error "Thresholds file not found: $THRESHOLDS_FILE"
        exit 1
    fi
    
    # Check if model directory exists
    if [ ! -d "$MODEL_DIR" ]; then
        log_warn "Model directory not found: $MODEL_DIR"
    fi
    
    log_info "Environment setup complete"
    log_info "Thresholds file: $THRESHOLDS_FILE"
    log_info "Model directory: $MODEL_DIR"
}

#===============================================================================
# ANALYSIS FUNCTIONS
#===============================================================================

# Run analysis for a dataset
run_analysis() {
    local dataset=$1
    local input_file=$2
    local output_dir=$3
    local thresholds_file=$4
    local model_dir=$5
    
    log_info "Processing dataset: $dataset"
    log_info "Input file: $input_file"
    log_info "Output directory: $output_dir"
    
    # Create output directory
    mkdir -p "$output_dir"
    check_status "Failed to create output directory for $dataset"
    
    # Run the analysis
    python "$SOURCE_DIR/formula/phase4.py" \
        --input-file "$input_file" \
        --output-dir "$output_dir" \
        --thresholds-file "$thresholds_file" \
        --model-dir "$model_dir" \
        --remove-outliers
    
    check_status "Metrics analysis failed for dataset: $dataset"
    
    log_info "Metrics analysis completed successfully for dataset: $dataset"
    return 0
}

# Process all datasets
process_all_datasets() {
    log_section "Starting Job Execution"
    
    start_time=$(date +%s)
    
    # Process each dataset
    for dataset in "${process_datasets[@]}"; do
        log_section "Processing Dataset: $dataset"
        
        # Define input and output paths
        input_file="$INPUT_BASE_DIR/$dataset/metrics_df.csv"
        output_dir="$OUTPUT_DIR/$dataset"
        
        # Check if input file exists
        if [ ! -f "$input_file" ]; then
            log_warn "Input file not found for dataset '$dataset': $input_file"
            log_warn "Skipping dataset: $dataset"
            continue
        fi
        
        # Run analysis
        run_analysis "$dataset" "$input_file" "$output_dir" "$THRESHOLDS_FILE" "$MODEL_DIR"
    done
    
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    hours=$((elapsed_time / 3600))
    minutes=$(( (elapsed_time % 3600) / 60 ))
    seconds=$((elapsed_time % 60))
    
    log_info "Total runtime: ${hours}h ${minutes}m ${seconds}s"
    return 0
}

# Report on completion
report_results() {
    log_section "Job Completed Successfully"
    
    for dataset in "${process_datasets[@]}"; do
        output_dir="$OUTPUT_DIR/$dataset"
        if [ -d "$output_dir" ]; then
            # Check for plots in the vis directory
            plot_count=$(find "$output_dir/vis" -name "*.png" 2>/dev/null | wc -l)
            log_info "Dataset: $dataset - Generated $plot_count plots"
        else
            log_warn "Dataset: $dataset - Output directory not found: $output_dir"
        fi
    done
    
    log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    # Parse command line arguments
    parse_arguments "$@"
    
    # Setup environment
    setup_environment
    
    # Process all datasets
    process_all_datasets
    
    # Report results
    report_results
    
    # Deactivate the virtual environment
    deactivate
}

# Execute main with all arguments
main "$@"

exit 0

#===============================================================================
# USAGE EXAMPLES (For reference only, not executed)
#===============================================================================

# Run aggregation for all datasets:
#   SLURM:  sbatch run.sh --datasets all
#   PYTHON: python "/data/home/djbf/bls/rq3/formula/phase4.py" \
#       --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase3/baseline/metrics_df.csv" 
#       --output-dir "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline" \
#       --thresholds-file "/data/home/djbf/bls/rq3/formula/thresholds/quantile.json" \
#       --model-dir "/data/home/djbf/storage/bls/rq1/outputs/phase4/combined_external_eval/stable16/experiments/penalty_elasticnet/models"
#

# Run aggregation for a specific dataset:
#   SLURM:  sbatch run.sh --datasets baseline
#   PYTHON: python "/data/home/djbf/bls/rq3/formula/phase4.py" \
#       --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase3/baseline/metrics_df.csv"
#       --output-dir "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline" \
#       --thresholds-file "/data/home/djbf/bls/rq3/formula/thresholds/quantile.json" \
#       --model-dir "/data/home/djbf/storage/bls/rq1/outputs/phase4/combined_external_eval/stable16/experiments/penalty_elasticnet/models"

# Run aggregation for combined datasets:
#   SLURM:  sbatch /data/home/djbf/bls/rq3/formula/run.sh --datasets baseline+fewshot+finetuned+finetunedIt+claude
#   PYTHON: python "/data/home/djbf/bls/rq3/formula/phase4.py" \
#       --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase3/baseline+fewshot+finetuned+finetunedIt+claude/metrics_df.csv" \
#       --output-dir "/data/home/djbf/storage/bls/rq3/outputs/phase4/baseline+fewshot+finetuned+finetunedIt+claude" \
#       --thresholds-file "/data/home/djbf/bls/rq3/formula/thresholds/quantile.json" \
#       --model-dir "/data/home/djbf/storage/bls/rq1/outputs/phase4/combined_external_eval/stable16/experiments/penalty_elasticnet/models"