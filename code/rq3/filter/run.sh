#!/bin/bash
#SBATCH --job-name=phase3_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --chdir=/data/home/djbf/storage/bls/rq3
#SBATCH --output=/data/home/djbf/storage/bls/rq3/logs/%j/phase3_analysis.out
#SBATCH --error=/data/home/djbf/storage/bls/rq3/logs/%j/phase3_analysis.err
#SBATCH --time=8:00:00

#===============================================================================
# Biomedical Language Simplification (BLS) - Phase 3 Analysis
# This script runs the phase 3 analysis for extracting and normalizing metrics
#===============================================================================

# Global configuration
readonly HOME_DIR="/data/home/djbf"
readonly SOURCE_DIR="$HOME_DIR/bls/rq3"
readonly BASE_DIR="$HOME_DIR/storage/bls"
readonly OUTPUT_DIR="$BASE_DIR/rq3/outputs"
readonly AVAILABLE_DATASETS=("baseline" "fewshot" "finetuned" "finetunedIt" "claude" "baseline+fewshot+finetuned+finetunedIt+claude" "all")

# Global variables
DATASETS=""
LOGS_DIR=""
declare -a process_datasets

#===============================================================================
# LOGGING FUNCTIONS
#===============================================================================

log_section() {
    echo -e "\n\033[1;36m=== $1 ===\033[0m\n"
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
    if [[ $? -ne 0 ]]; then
        log_error "$1"
        exit 1
    fi
}

#===============================================================================
# UTILITY FUNCTIONS
#===============================================================================

show_usage() {
    cat << EOF
Usage: sbatch run.sh [OPTIONS]

Options:
  --datasets DATASETS      Datasets to analyze, comma-separated
                           (baseline,fewshot,finetuned,finetunedIt,claude,
                           baseline+fewshot+finetuned+finetunedIt+claude)
                           or 'all' for all datasets

Examples:
  sbatch run.sh --datasets all
  sbatch run.sh --datasets baseline,fewshot
  sbatch run.sh --datasets baseline+fewshot+finetuned+finetunedIt+claude
EOF
    exit 1
}

is_valid_dataset() {
    local dataset="$1"
    for available in "${AVAILABLE_DATASETS[@]}"; do
        [[ "$dataset" == "$available" ]] && return 0
    done
    return 1
}

#===============================================================================
# SETUP FUNCTIONS
#===============================================================================

parse_arguments() {
    log_section "Parsing Arguments"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --datasets) DATASETS="$2"; shift 2 ;;
            --help|-h) show_usage ;;
            *) log_error "Unknown argument: $1"; show_usage ;;
        esac
    done
    
    [[ -z "$DATASETS" ]] && { log_error "Required argument: --datasets"; show_usage; }
    
    process_dataset_list
}

process_dataset_list() {
    if [[ "$DATASETS" == "all" ]]; then
        process_datasets=("${AVAILABLE_DATASETS[@]}")
        log_info "Datasets to process: ${process_datasets[*]}"
        return
    fi
    
    IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
    
    for dataset in "${DATASET_LIST[@]}"; do
        if is_valid_dataset "$dataset"; then
            process_datasets+=("$dataset")
            continue
        fi
        log_warn "Dataset '$dataset' is not recognized. Valid datasets: ${AVAILABLE_DATASETS[*]}"
    done
    
    if [[ ${#process_datasets[@]} -eq 0 ]]; then
        log_error "No valid datasets specified"
        exit 1
    fi
    
    log_info "Datasets to process: ${process_datasets[*]}"
}

setup_environment() {
    log_section "Environment Setup"
    
    LOGS_DIR="$BASE_DIR/rq3/logs/$SLURM_JOB_ID"
    
    mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"
    check_status "Failed to create output directories"
    
    log_info "Loading required modules..."
    module load python
    check_status "Failed to load Python module"
    
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    check_status "Failed to activate virtual environment"
    
    log_info "Environment setup complete"
}

#===============================================================================
# ANALYSIS FUNCTIONS
#===============================================================================

run_phase3_analysis() {
    local dataset="$1"
    local input_file="$OUTPUT_DIR/phase2/$dataset/aggregated_metrics.json"
    local output_dir="$OUTPUT_DIR/phase3/$dataset"
    
    log_info "Running Phase 3 analysis for dataset: $dataset"
    log_info "Input: $input_file"
    log_info "Output: $output_dir"
    
    [[ ! -f "$input_file" ]] && { log_error "Input file does not exist: $input_file"; return 1; }
    
    mkdir -p "$output_dir"
    check_status "Failed to create output directory for $dataset"
    
    python "$SOURCE_DIR/filter/phase3.py" \
        --input-file "$input_file" \
        --output-dir "$output_dir"
    
    check_status "Phase 3 analysis failed for dataset: $dataset"
    log_info "Phase 3 analysis completed successfully for dataset: $dataset"
}

process_all_datasets() {
    log_section "Starting Job Execution"
    
    local start_time=$(date +%s)
    
    for dataset in "${process_datasets[@]}"; do
        log_section "Processing Dataset: $dataset"
        run_phase3_analysis "$dataset"
    done
    
    calculate_runtime "$start_time"
}

calculate_runtime() {
    local start_time="$1"
    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))
    local hours=$((elapsed_time / 3600))
    local minutes=$(( (elapsed_time % 3600) / 60 ))
    local seconds=$((elapsed_time % 60))
    
    log_info "Total runtime: ${hours}h ${minutes}m ${seconds}s"
}

report_results() {
    log_section "Job Completed Successfully"
    
    for dataset in "${process_datasets[@]}"; do
        local output_dir="$OUTPUT_DIR/phase3/$dataset"
        local metrics_file="$output_dir/metrics_df.csv"
        
        if [[ -f "$metrics_file" ]]; then
            local file_size=$(du -h "$metrics_file" | cut -f1)
            local row_count=$(tail -n +2 "$metrics_file" | wc -l)
            log_info "Dataset: $dataset - Metrics file: $metrics_file (Size: $file_size, Rows: $row_count)"
        else
            log_warn "Dataset: $dataset - Output metrics file not found: $metrics_file"
        fi
    done
    
    log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

main() {
    parse_arguments "$@"
    setup_environment
    process_all_datasets
    report_results
    deactivate
}

# Execute main with all arguments
main "$@"

#===============================================================================
# USAGE EXAMPLES (For reference only, not executed)
#===============================================================================

# Run aggregation for all datasets:
#   SLURM:  sbatch run.sh --datasets all
#   PYTHON: python "/data/home/djbf/bls/rq3/filter/phase3.py" \
#       --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase2/baseline+fewshot+finetuned+finetunedIt+claude/aggregated_metrics.json" \
#       --output-dir "/data/home/djbf/storage/bls/rq3/outputs/phase3/baseline+fewshot+finetuned+finetunedIt+claude"

# Run aggregation for a specific dataset:
#   SLURM:  sbatch run.sh --datasets baseline
#   PYTHON: python "/data/home/djbf/bls/rq3/filter/phase3.py" \
#       --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase2/baseline/aggregated_metrics.json" \
#       --output-dir "/data/home/djbf/storage/bls/rq3/outputs/phase3/baseline"

# Run aggregation for combined datasets:
#   SLURM:  sbatch /data/home/djbf/bls/rq3/filter/run.sh --datasets baseline+fewshot+finetuned+finetunedIt+claude
#   PYTHON: python "/data/home/djbf/bls/rq3/filter/phase3.py" \
#       --input-file "/data/home/djbf/storage/bls/rq3/outputs/phase2/baseline+fewshot+finetuned+finetunedIt+claude/aggregated_metrics.json" \
#       --output-dir "/data/home/djbf/storage/bls/rq3/outputs/phase3/baseline+fewshot+finetuned+finetunedIt+claude"