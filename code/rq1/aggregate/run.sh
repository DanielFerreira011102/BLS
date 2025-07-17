#!/bin/bash
#SBATCH --job-name=metrics_agg      # Name of the job visible in queue
#SBATCH --nodes=1                   # Number of compute nodes to allocate
#SBATCH --ntasks=1                  # Number of tasks (processes) to create
#SBATCH --cpus-per-task=4           # CPU cores per task
#SBATCH --mem=16G                   # Memory allocation per node
#SBATCH --partition=cpu             # Compute partition/queue to use
#SBATCH --nodelist=cpu-srv-01       # Specific node to run on (can be overridden)
#SBATCH --chdir=/data/home/djbf/storage/bls/rq1  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq1/logs/%j/metrics_agg.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq1/logs/%j/metrics_agg.err   # Standard error file (%j = job ID)
#SBATCH --time=4:00:00              # Time limit (4 hours)

#===============================================================================
# Biomedical Language Simplification (BLS) - Metrics Aggregator
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
    echo "Usage: sbatch run.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --datasets DATASETS      Datasets to aggregate, comma-separated"
    echo "                           (claude,cochrane,claude+cochrane,plaba,plaba-sentence,plaba-paragraph,claude+cochrane+plaba)"
    echo "                           or 'all' for all datasets"
    echo "  --input-dir DIR          Base directory containing input JSON files (default: /data/home/djbf/storage/bls/rq1/outputs)"
    echo "  --exclude PATTERN        Glob pattern to exclude files (default: \"**/deprecated/**/*.json\")"
    echo ""
    echo "Examples:"
    echo "  sbatch run.sh --datasets all"
    echo "  sbatch run.sh --datasets claude,cochrane"
    echo "  sbatch run.sh --datasets plaba --exclude \"**/test/**/*.json\""
    exit 1
}

# Run aggregation for a dataset
run_aggregation() {
    local dataset=$1
    local input_dir=$2
    local exclude_pattern=$3
    local output_file="$OUTPUT_DIR/phase0/$dataset/aggregated_metrics.json"
    local pattern=""
    
    # Determine pattern based on dataset
    case "$dataset" in
        "claude") pattern="phase0/claude/**/readability_metrics.json" ;;
        "cochrane") pattern="phase0/cochrane/**/readability_metrics.json" ;;
        "claude+cochrane") pattern="phase0/c*/**/readability_metrics.json" ;;
        "claude+cochrane+plaba") pattern="phase0/**/readability_metrics.json" ;;
        "plaba") pattern="phase0/plaba*/**/readability_metrics.json" ;;
        "plaba-sentence") pattern="phase0/plaba-sentence/**/readability_metrics.json" ;;
        "plaba-paragraph") pattern="phase0/plaba-paragraph/**/readability_metrics.json" ;;
        *) log_error "Unknown dataset: $dataset"; return 1 ;;
    esac
    
    log_info "Aggregating metrics for dataset: $dataset"
    log_info "Pattern: $pattern"
    log_info "Exclude: $exclude_pattern"
    log_info "Output: $output_file"
    
    # Create output directory
    mkdir -p "$(dirname "$output_file")"
    check_status "Failed to create output directory for $dataset"
    
    # Run the aggregation
    python "$SOURCE_DIR/aggregate/metrics_aggregator.py" \
        --input-dir "$input_dir" \
        --output-file "$output_file" \
        --pattern "$pattern" \
        --exclude "$exclude_pattern"
    
    check_status "Metrics aggregation failed for dataset: $dataset"
    
    log_info "Metrics aggregation completed successfully for dataset: $dataset"
}

#===============================================================================
# DIRECTORY SETUP
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq1"
BASE_DIR="$HOME_DIR/storage/bls"
OUTPUT_DIR="$BASE_DIR/rq1/outputs"
LOGS_DIR="$BASE_DIR/rq1/logs/$SLURM_JOB_ID"

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
AVAILABLE_DATASETS=("claude" "cochrane" "claude+cochrane" "plaba" "plaba-sentence" "plaba-paragraph" "claude+cochrane+plaba")

# Determine which datasets to process
process_datasets=()

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
    output_file="$OUTPUT_DIR/phase0/$dataset/aggregated_metrics.json"
    if [ -f "$output_file" ]; then
        file_size=$(du -h "$output_file" | cut -f1)
        log_info "Dataset: $dataset - Aggregated metrics file: $output_file (Size: $file_size)"
    else
        log_warn "Dataset: $dataset - Output file not found: $output_file"
    fi
done

log_info "Job ID: $SLURM_JOB_ID completed at $(date)"

# Deactivate the virtual environment
deactivate

#===============================================================================
# USAGE EXAMPLES (For reference only, not executed)
#===============================================================================

# Run aggregation for all datasets:
#   SLURM:  sbatch run.sh --datasets all
#   PYTHON: python "/data/home/djbf/bls/rq1/metrics/metrics_aggregator.py" \
#           --input-dir "/data/home/djbf/storage/bls/rq1/outputs/" \
#           --output-file "/data/home/djbf/storage/bls/rq1/outputs/phase0/claude/aggregated_metrics.json" \
#           --pattern "phase0/claude/**/readability_metrics.json" \
#           --exclude "**/deprecated/**/*.json"
#           python "/data/home/djbf/bls/rq1/metrics/metrics_aggregator.py" \
#           --input-dir "/data/home/djbf/storage/bls/rq1/outputs/" \
#           --output-file "/data/home/djbf/storage/bls/rq1/outputs/phase0/cochrane/aggregated_metrics.json" \
#           --pattern "phase0/cochrane/**/readability_metrics.json" \
#           --exclude "**/deprecated/**/*.json"
#           # ... repeat for all other datasets

# Run aggregation for specific datasets:
#   SLURM:  sbatch run.sh --datasets claude,cochrane
#   PYTHON: python "/data/home/djbf/bls/rq1/metrics/metrics_aggregator.py" \
#           --input-dir "/data/home/djbf/storage/bls/rq1/outputs/" \
#           --output-file "/data/home/djbf/storage/bls/rq1/outputs/phase0/claude/aggregated_metrics.json" \
#           --pattern "phase0/claude/**/readability_metrics.json" \
#           --exclude "**/deprecated/**/*.json"
#           python "/data/home/djbf/bls/rq1/metrics/metrics_aggregator.py" \
#           --input-dir "/data/home/djbf/storage/bls/rq1/outputs/" \
#           --output-file "/data/home/djbf/storage/bls/rq1/outputs/phase0/cochrane/aggregated_metrics.json" \
#           --pattern "phase0/cochrane/**/readability_metrics.json" \
#           --exclude "**/deprecated/**/*.json"

# Run aggregation with custom input directory:
#   SLURM:  sbatch run.sh --datasets plaba --input-dir "/data/home/djbf/storage/bls/other/outputs"
#   PYTHON: python "/data/home/djbf/bls/rq1/metrics/metrics_aggregator.py" \
#           --input-dir "/data/home/djbf/storage/bls/other/outputs" \
#           --output-file "/data/home/djbf/storage/bls/rq1/outputs/phase0/plaba/aggregated_metrics.json" \
#           --pattern "phase0/plaba*/**/readability_metrics.json" \
#           --exclude "**/deprecated/**/*.json"

# Run aggregation with custom exclude pattern:
#   SLURM:  sbatch run.sh --datasets claude --exclude "**/test/**/*.json"
#   PYTHON: python "/data/home/djbf/bls/rq1/metrics/metrics_aggregator.py" \
#           --input-dir "/data/home/djbf/storage/bls/rq1/outputs/" \
#           --output-file "/data/home/djbf/storage/bls/rq1/outputs/phase0/claude/aggregated_metrics.json" \
#           --pattern "phase0/claude/**/readability_metrics.json" \
#           --exclude "**/test/**/*.json"