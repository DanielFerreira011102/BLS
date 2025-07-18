#!/bin/bash
#SBATCH --job-name=rq2_metrics_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --nodelist=cpu-srv-02
#SBATCH --chdir=/data/home/djbf/storage/bls/rq2
#SBATCH --output=/data/home/djbf/storage/bls/rq2/logs/%j/metrics_agg.out
#SBATCH --error=/data/home/djbf/storage/bls/rq2/logs/%j/metrics_agg.err
#SBATCH --time=4:00:00

#===============================================================================
# Biomedical Language Simplification (BLS) - RQ2 Metrics Aggregator
# This script aggregates readability metrics from multiple JSON files
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
    cat << EOF
Usage: sbatch run.sh [OPTIONS]

Options:
  --datasets DATASETS      Datasets to aggregate, comma-separated
                           (bioasq,liveqa,medicationqa,mediqaans,medquad,
                           liveqa+medicationqa+mediqaans+bioasq+medquad)
                           or 'all' for all datasets individually
  --input-dir DIR          Base directory containing input JSON files (default: $OUTPUT_DIR)
  --exclude PATTERN        Glob pattern to exclude files (default: "**/deprecated/**/*.json")

Examples:
  sbatch run.sh --datasets all
  sbatch run.sh --datasets bioasq,liveqa
  sbatch run.sh --datasets liveqa+medicationqa+mediqaans+bioasq+medquad
  sbatch run.sh --datasets medquad --exclude "**/test/**/*.json"
EOF
    exit 1
}

# Get pattern for dataset
get_pattern() {
    case "$1" in
        "bioasq") echo "phase1/bioasq/**/**/readability_metrics.json" ;;
        "liveqa") echo "phase1/liveqa/**/**/readability_metrics.json" ;;
        "medicationqa") echo "phase1/medicationqa/**/**/readability_metrics.json" ;;
        "mediqaans") echo "phase1/mediqaans/**/**/readability_metrics.json" ;;
        "medquad") echo "phase1/medquad/**/**/readability_metrics.json" ;;
        "liveqa+medicationqa+mediqaans+bioasq+medquad") echo "phase1/**/**/readability_metrics.json" ;;
        *) log_error "Unknown dataset: $1"; return 1 ;;
    esac
}

# Run aggregation for a dataset
run_aggregation() {
    local dataset=$1 input_dir=$2 exclude_pattern=$3
    local output_file="$OUTPUT_DIR/phase1/$dataset/aggregated_metrics.json"
    local pattern
    
    pattern=$(get_pattern "$dataset") || return 1
    
    log_info "Aggregating metrics for dataset: $dataset"
    log_info "Pattern: $pattern"
    log_info "Output: $output_file"
    
    mkdir -p "$(dirname "$output_file")"
    check_status "Failed to create output directory for $dataset"
    
    python "$SOURCE_DIR/aggregate/metrics_aggregator.py" \
        --input-dir "$input_dir" \
        --output-file "$output_file" \
        --pattern "$pattern" \
        --exclude "$exclude_pattern"
    
    check_status "Metrics aggregation failed for dataset: $dataset"
    log_info "Completed successfully for dataset: $dataset"
}

#===============================================================================
# DIRECTORY SETUP
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq2"
OUTPUT_DIR="$HOME_DIR/storage/bls/rq2/outputs"
LOGS_DIR="$HOME_DIR/storage/bls/rq2/logs/$SLURM_JOB_ID"

# Create required directories
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

# Activate virtual environment
log_info "Activating virtual environment..."
source "$HOME_DIR/bls/venv/bin/activate"
check_status "Failed to activate virtual environment"

#===============================================================================
# ARGUMENT PARSING
#===============================================================================

log_section "Argument Parsing"

# Default values
DATASETS=""
INPUT_DIR="$OUTPUT_DIR"
EXCLUDE_PATTERN="**/deprecated/**/*.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --datasets) DATASETS="$2"; shift 2 ;;
        --input-dir) INPUT_DIR="$2"; shift 2 ;;
        --exclude) EXCLUDE_PATTERN="$2"; shift 2 ;;
        --help|-h) show_usage ;;
        *) log_error "Unknown argument: $1"; show_usage ;;
    esac
done

# Validate required arguments
if [ -z "$DATASETS" ]; then
    log_error "Required argument: --datasets"
    show_usage
fi

# Define available datasets
AVAILABLE_DATASETS=("bioasq" "liveqa" "medicationqa" "mediqaans" "medquad" "liveqa+medicationqa+mediqaans+bioasq+medquad")

# Parse datasets to process
if [ "$DATASETS" == "all" ]; then
    process_datasets=("bioasq" "liveqa" "medicationqa" "mediqaans" "medquad")
else
    IFS=',' read -ra process_datasets <<< "$DATASETS"
fi

#===============================================================================
# EXECUTION
#===============================================================================

log_section "Starting Job Execution"

log_info "Datasets to process: ${process_datasets[*]}"
log_info "Input directory: $INPUT_DIR"
log_info "Exclude pattern: $EXCLUDE_PATTERN"

start_time=$(date +%s)

# Process each dataset
for dataset in "${process_datasets[@]}"; do
    log_section "Processing Dataset: $dataset"
    run_aggregation "$dataset" "$INPUT_DIR" "$EXCLUDE_PATTERN"
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(( (elapsed_time % 3600) / 60 ))
seconds=$((elapsed_time % 60))

#===============================================================================
# COMPLETION
#===============================================================================

log_section "Job Completed Successfully"

log_info "Total runtime: ${hours}h ${minutes}m ${seconds}s"

for dataset in "${process_datasets[@]}"; do
    output_file="$OUTPUT_DIR/phase1/$dataset/aggregated_metrics.json"
    if [ -f "$output_file" ]; then
        file_size=$(du -h "$output_file" | cut -f1)
        log_info "Dataset: $dataset - Output: $output_file (Size: $file_size)"
    else
        log_warn "Dataset: $dataset - Output file not found: $output_file"
    fi
done

log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
deactivate