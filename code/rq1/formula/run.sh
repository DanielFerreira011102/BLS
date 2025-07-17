#!/bin/bash
#SBATCH --job-name=bls_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=30G
#SBATCH --partition=cpu
#SBATCH --chdir=/data/home/djbf/storage/bls/rq1
#SBATCH --output=/data/home/djbf/storage/bls/rq1/logs/%j/all_phases.out
#SBATCH --error=/data/home/djbf/storage/bls/rq1/logs/%j/all_phases.err
#SBATCH --time=48:00:00

#===============================================================================
# BIOMEDICAL LANGUAGE SIMPLIFICATION (BLS) - ALL PHASES PIPELINE
#
# This script runs all phases (1-4) of the complexity metrics pipeline
# with simple configuration options through the --sets argument.
#===============================================================================

#===============================================================================
# 1. CONFIGURATION
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq1"
BASE_DIR="$HOME_DIR/storage/bls"
OUTPUT_DIR="$BASE_DIR/rq1/outputs"
LOGS_DIR="$BASE_DIR/rq1/logs/$SLURM_JOB_ID"
FEATURE_SETS_DIR="$SOURCE_DIR/formula/feature_sets"

# Default normalization and parameters
NORMALIZATION="z_score"
FEATURES_SET=""  # Default empty, will use phase3 results or automatic selection

# Phase 3 options
USE_ALL_DATA=false
FILTER_BY_GROUP=false
CORR_THRESHOLD=""

# Define valid sets
VALID_SETS=(
    "all"
    "phase1_all"
    "phase2_all"
    
    # Phase 3 - Claude only
    "phase3_claude_no_eval"
    "phase3_claude_external_eval"
    "phase3_claude_all_eval"
    
    # Phase 3 - Combined dataset
    "phase3_combined_no_eval"
    "phase3_combined_external_eval"
    "phase3_combined_all_eval"
    
    # Phase 4 - Claude only
    "phase4_claude_no_eval"
    "phase4_claude_external_eval"
    "phase4_claude_all_eval"
    
    # Phase 4 - Combined dataset
    "phase4_combined_no_eval"
    "phase4_combined_external_eval"
    "phase4_combined_all_eval"
)

# Default set
SETS="all"

# Define datasets
ALL_DATASETS=("claude" "cochrane" "plaba-sentence" "plaba-paragraph")
COMBINED_DATASET="claude+cochrane+plaba-sentence+plaba-paragraph"
ALL_EVAL_DATASETS="claude+cochrane+plaba-sentence+plaba-paragraph"

#===============================================================================
# 2. HELPER FUNCTIONS
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

# Check if element is in array
contains() {
    local element=$1
    shift
    for e in "$@"; do
        [[ "$e" == "$element" ]] && return 0
    done
    return 1
}

# Generate a suffix based on parameter combinations
get_param_suffix() {
    local suffix=""
    
    if [ "$USE_ALL_DATA" = true ]; then
        suffix="${suffix}_corr_all"
    fi
    
    if [ "$FILTER_BY_GROUP" = true ]; then
        suffix="${suffix}+group"
    fi
    
    if [ -n "$CORR_THRESHOLD" ]; then
        # Remove decimal point for cleaner directory names
        local threshold_str=$(echo "$CORR_THRESHOLD" | tr -d '.')
        suffix="${suffix}+${threshold_str}"
    fi
    
    echo "$suffix"
}

# Function to get external datasets (datasets not used in training)
get_external_datasets() {
    local training_datasets="$1"
    local external_datasets=""
    
    # Separate training datasets into array
    IFS='+' read -ra TRAIN_DS <<< "$training_datasets"
    
    # Check each dataset to see if it's in the training set
    for ds in "${ALL_DATASETS[@]}"; do
        if ! contains "$ds" "${TRAIN_DS[@]}"; then
            if [ -z "$external_datasets" ]; then
                external_datasets="$ds"
            else
                external_datasets="$external_datasets+$ds"
            fi
        fi
    done
    
    echo "$external_datasets"
}

# Show usage information
show_usage() {
    cat << EOF
Usage: sbatch run.sh [OPTIONS]

Options:
  --sets SETS    Comma-separated list of sets to run
                 Available:
                   all                          - Run all sets
                   phase1_all                   - Run Phase 1 for all datasets
                   phase2_all                   - Run Phase 2 for all datasets
                   
                   # Phase 3 - Claude only
                   phase3_claude_no_eval        - Run Phase 3 for Claude without evaluation
                   phase3_claude_external_eval  - Run Phase 3 for Claude with external evaluation
                   phase3_claude_all_eval       - Run Phase 3 for Claude with all evaluation
                   
                   # Phase 3 - Combined dataset
                   phase3_combined_no_eval      - Run Phase 3 for combined without evaluation
                   phase3_combined_external_eval - Run Phase 3 for combined with external evaluation
                   phase3_combined_all_eval     - Run Phase 3 for combined with all evaluation
                   
                   # Phase 4 - Claude only
                   phase4_claude_no_eval        - Run Phase 4 for Claude without evaluation
                   phase4_claude_external_eval  - Run Phase 4 for Claude with external evaluation
                   phase4_claude_all_eval       - Run Phase 4 for Claude with all evaluation
                   
                   # Phase 4 - Combined dataset
                   phase4_combined_no_eval      - Run Phase 4 for combined without evaluation
                   phase4_combined_external_eval - Run Phase 4 for combined with external evaluation
                   phase4_combined_all_eval     - Run Phase 4 for combined with all evaluation
                   
                 Default: all
  --features SET Specify a feature set to use for Phase 4 (e.g., core7, lasso13)
                 Feature set must exist in $FEATURE_SETS_DIR/
  --use-all-data          Enable --use-all-data-for-correlation in phase 3
  --filter-by-group       Enable --filter-correlation-by-group in phase 3
  --corr-threshold VALUE  Set the correlation threshold for phase 3
  --help|-h      Show this help message
EOF
    exit 1
}

#===============================================================================
# 3. ENVIRONMENT SETUP
#===============================================================================

setup_environment() {
    log_section "Environment Setup"
    
    # Create logs directory for this job
    mkdir -p "$LOGS_DIR"
    if [ $? -ne 0 ]; then
        log_error "Failed to create logs directory"
        exit 1
    fi
    
    # Load required modules
    log_info "Loading required modules..."
    module load python
    if [ $? -ne 0 ]; then
        log_error "Failed to load Python module"
        exit 1
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    if [ $? -ne 0 ]; then
        log_error "Failed to activate virtual environment"
        exit 1
    fi
}

#===============================================================================
# 4. ARGUMENT PARSING
#===============================================================================

parse_arguments() {
    log_section "Argument Parsing"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --sets) SETS="$2"; shift 2 ;;
            --features) FEATURES_SET="$2"; shift 2 ;;
            --use-all-data) USE_ALL_DATA=true; shift ;;
            --filter-by-group) FILTER_BY_GROUP=true; shift ;;
            --corr-threshold) CORR_THRESHOLD="$2"; shift 2 ;;
            --help|-h) show_usage ;;
            *) log_error "Unknown argument: $1"; show_usage ;;
        esac
    done
    
    # Process SETS
    if [ "$SETS" == "all" ]; then
        SET_LIST=("${VALID_SETS[@]}")
        # Remove "all" from the list to avoid recursion
        SET_LIST=("${SET_LIST[@]/all}")
    else
        IFS=',' read -ra SET_LIST <<< "$SETS"
    fi
    
    # Validate sets
    for set_name in "${SET_LIST[@]}"; do
        if ! contains "$set_name" "${VALID_SETS[@]}"; then
            log_error "Invalid set name: $set_name"
            show_usage
        fi
    done
    
    # Check if features file exists if specified
    if [ -n "$FEATURES_SET" ]; then
        local features_file="${FEATURE_SETS_DIR}/${FEATURES_SET}.json"
        if [ ! -f "$features_file" ]; then
            log_error "Features file not found: $features_file"
            exit 1
        else
            log_info "Using custom features file: $features_file"
        fi
    fi
    
    # Get parameter suffix for output directories
    PARAM_SUFFIX=$(get_param_suffix)
    
    # Log configuration
    log_info "Sets to run: ${SET_LIST[*]}"
    if [ -n "$FEATURES_SET" ]; then
        log_info "Features set: $FEATURES_SET"
    fi
    if [ "$USE_ALL_DATA" = true ]; then
        log_info "Use all data for correlation: enabled"
    else
        log_info "Use all data for correlation: disabled"
    fi
    if [ "$FILTER_BY_GROUP" = true ]; then
        log_info "Filter correlation by group: enabled"
    else
        log_info "Filter correlation by group: disabled"
    fi
    if [ -n "$CORR_THRESHOLD" ]; then
        log_info "Correlation threshold: $CORR_THRESHOLD"
    else
        log_info "Correlation threshold: default"
    fi
    if [ -n "$PARAM_SUFFIX" ]; then
        log_info "Parameter suffix for output directories: $PARAM_SUFFIX"
    fi
}

#===============================================================================
# 5. PHASE 1 FUNCTIONS
#===============================================================================

run_phase1_for_dataset() {
    local dataset="$1"
    local log_dir="$LOGS_DIR/phase1_${dataset}"
    
    log_section "Running Phase 1 for dataset: $dataset"
    
    mkdir -p "$log_dir"
    local input_file="$OUTPUT_DIR/phase0/$dataset/aggregated_metrics.json"
    local output_dir="$OUTPUT_DIR/phase1/$dataset"

    if [ ! -f "$input_file" ]; then
        log_error "Input file for Phase 1 not found: $input_file"
        return 1
    fi

    mkdir -p "$output_dir"

    # Standard problematic features to ignore
    local ignore_features=(
        "commonlit/deberta_ensemble/fold_1"
        "commonlit/deberta_ensemble/fold_2"
        "commonlit/deberta_ensemble/fold_3"
        "commonlit/deberta_ensemble/fold_4"
        "commonlit/deberta_ensemble/fold_5"
        "commonlit/deberta_ensemble/fold_6"
        "commonlit/deberta_ensemble/ensemble"
        "llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/dimension_scores/cognitive_load"
        "llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/overall_score"
        "llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/dimension_scores/background_knowledge"
        "llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/dimension_scores/conceptual_density"
        "llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/dimension_scores/vocabulary_complexity"
        "llm/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind/dimension_scores/syntactic_complexity"
    )

    local cmd="python $SOURCE_DIR/formula/phase1.py --input-file \"$input_file\" --output-dir \"$output_dir\" --ignore-features"
    for feature in "${ignore_features[@]}"; do
        cmd="$cmd \"$feature\""
    done

    log_info "Executing: $cmd"
    eval "$cmd" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
    local status=$?

    if [ $status -eq 0 ]; then
        log_info "Phase 1 completed successfully for dataset: $dataset"
    else
        log_error "Phase 1 failed with exit code $status for dataset: $dataset"
    fi

    return $status
}

run_phase1_all() {
    log_section "Running Phase 1 for all datasets"
    
    for dataset in "${ALL_DATASETS[@]}"; do
        run_phase1_for_dataset "$dataset"
    done
    
    # Also run Phase 1 for the combined dataset if needed
    run_phase1_for_dataset "$COMBINED_DATASET"
}

#===============================================================================
# 6. PHASE 2 FUNCTIONS
#===============================================================================

run_phase2_for_dataset() {
    local dataset="$1"
    local log_dir="$LOGS_DIR/phase2_${dataset}"
    
    log_section "Running Phase 2 for dataset: $dataset"
    
    mkdir -p "$log_dir"
    local phase1_dir="$OUTPUT_DIR/phase1/$dataset"
    local output_dir="$OUTPUT_DIR/phase2/$dataset"

    if [ ! -d "$phase1_dir" ]; then
        log_error "Phase 1 directory not found: $phase1_dir"
        return 1
    fi

    mkdir -p "$output_dir"

    local cmd="python $SOURCE_DIR/formula/phase2.py --phase1-dir \"$phase1_dir\" --output-dir \"$output_dir\""
    
    log_info "Executing: $cmd"
    eval "$cmd" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
    local status=$?

    if [ $status -eq 0 ]; then
        log_info "Phase 2 completed successfully for dataset: $dataset"
    else
        log_error "Phase 2 failed with exit code $status for dataset: $dataset"
    fi

    return $status
}

run_phase2_all() {
    log_section "Running Phase 2 for all datasets"
    
    for dataset in "${ALL_DATASETS[@]}"; do
        run_phase2_for_dataset "$dataset"
    done
    
    # Also run Phase 2 for the combined dataset if needed
    run_phase2_for_dataset "$COMBINED_DATASET"
}

#===============================================================================
# 7. PHASE 3 FUNCTIONS (Feature Selection)
#===============================================================================

run_phase3() {
    local dataset="$1"
    local eval_datasets="$2"
    local eval_mode="$3"  # "none", "external", or "all"
    
    # Create directory name with the parameter suffix
    local dir_suffix="${dataset}_${eval_mode}_eval${PARAM_SUFFIX}"
    local log_dir="$LOGS_DIR/phase3_${dir_suffix}"
    
    log_section "Running Phase 3 for dataset: $dataset"
    log_info "Evaluation mode: $eval_mode"
    
    if [ "$eval_mode" != "none" ]; then
        log_info "Evaluation datasets: $eval_datasets"
    else
        log_info "No evaluation datasets"
    fi
    
    mkdir -p "$log_dir"
    
    local phase1_dir="$OUTPUT_DIR/phase1"
    local output_dir="$OUTPUT_DIR/phase3/${dir_suffix}"
    
    mkdir -p "$output_dir"
    
    # Build command
    local cmd="python $SOURCE_DIR/formula/phase3.py \
      --phase1-dir \"$phase1_dir\" \
      --output-dir \"$output_dir\" \
      --training-datasets"
      
    # Add training datasets
    IFS='+' read -ra TRAIN_DS <<< "$dataset"
    for ds in "${TRAIN_DS[@]}"; do
        cmd="$cmd \"$ds\""
    done
    
    cmd="$cmd --normalization \"$NORMALIZATION\""
    
    # Add evaluation datasets ONLY if not in "none" mode
    if [ "$eval_mode" != "none" ] && [ -n "$eval_datasets" ]; then
        cmd="$cmd --evaluation-datasets"
        IFS='+' read -ra EVAL_DS <<< "$eval_datasets"
        for ds in "${EVAL_DS[@]}"; do
            cmd="$cmd \"$ds\""
        done
    fi
    
    # Add phase 3 specific options
    if [ "$USE_ALL_DATA" = true ]; then
        cmd="$cmd --use-all-data-for-correlation"
    fi
    if [ "$FILTER_BY_GROUP" = true ]; then
        cmd="$cmd --filter-correlation-by-group"
    fi
    if [ -n "$CORR_THRESHOLD" ]; then
        cmd="$cmd --correlation-threshold \"$CORR_THRESHOLD\""
    fi
    
    log_info "Executing: $cmd"
    eval "$cmd" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
    
    local status=$?
    if [ $status -eq 0 ]; then
        log_info "Phase 3 completed successfully for dataset: $dataset"
        grep -A 10 "Feature Selection" "$log_dir/stdout.log" | head -n 10
    else
        log_error "Phase 3 failed with exit code $status for dataset: $dataset"
    fi
    
    return $status
}

# Phase 3 - Claude only
run_phase3_claude_no_eval() {
    run_phase3 "claude" "" "none"
}

run_phase3_claude_external_eval() {
    local external_datasets=$(get_external_datasets "claude")
    run_phase3 "claude" "$external_datasets" "external"
}

run_phase3_claude_all_eval() {
    run_phase3 "claude" "$ALL_EVAL_DATASETS" "all"
}

# Phase 3 - Combined dataset
run_phase3_combined_no_eval() {
    run_phase3 "$COMBINED_DATASET" "" "none"
}

run_phase3_combined_external_eval() {
    local external_datasets=$(get_external_datasets "$COMBINED_DATASET")
    run_phase3 "$COMBINED_DATASET" "$external_datasets" "external"
}

run_phase3_combined_all_eval() {
    run_phase3 "$COMBINED_DATASET" "$ALL_EVAL_DATASETS" "all"
}

#===============================================================================
# 8. PHASE 4 FUNCTIONS (Train & Evaluate Formula)
#===============================================================================

run_phase4() {
    local dataset="$1"
    local eval_datasets="$2"
    local eval_mode="$3"  # "none", "external", or "all"
    local features_file="$4"  # Optional path to features file
    
    local features_name=""
    if [ -n "$features_file" ]; then
        features_name=$(basename "$features_file" .json)
    fi
    
    # Create directory name with parameter suffix
    local dir_suffix="${dataset}_${eval_mode}_eval${PARAM_SUFFIX}"
    if [ -n "$features_name" ]; then
        dir_suffix="${dir_suffix}_${features_name}"
    fi
    
    local log_dir="$LOGS_DIR/phase4_${dir_suffix}"
    
    log_section "Running Phase 4 for dataset: $dataset"
    log_info "Evaluation mode: $eval_mode"
    
    if [ "$eval_mode" != "none" ]; then
        log_info "Evaluation datasets: $eval_datasets"
    else
        log_info "No evaluation datasets"
    fi
    
    if [ -n "$features_file" ]; then
        log_info "Using features from: $features_file"
    fi
    
    mkdir -p "$log_dir"
    
    local phase1_dir="$OUTPUT_DIR/phase1"
    local output_dir="$OUTPUT_DIR/phase4/${dir_suffix}"
    
    mkdir -p "$output_dir"
    
    # Build command
    local cmd="python $SOURCE_DIR/formula/phase4.py \
      --phase1-dir \"$phase1_dir\" \
      --output-dir \"$output_dir\" \
      --training-datasets"
      
    # Add training datasets
    IFS='+' read -ra TRAIN_DS <<< "$dataset"
    for ds in "${TRAIN_DS[@]}"; do
        cmd="$cmd \"$ds\""
    done
    
    cmd="$cmd --normalization \"$NORMALIZATION\""

    # Add features file if specified
    if [ -n "$features_file" ]; then
        cmd="$cmd --features-file \"$features_file\""
    fi
    
    # Add evaluation datasets if not "none" AND we have evaluation datasets
    if [ "$eval_mode" != "none" ] && [ -n "$eval_datasets" ]; then
        cmd="$cmd --evaluation-datasets"
        IFS='+' read -ra EVAL_DS <<< "$eval_datasets"
        for ds in "${EVAL_DS[@]}"; do
            cmd="$cmd \"$ds\""
        done
    fi
    
    log_info "Executing: $cmd"
    eval "$cmd" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
    
    local status=$?
    if [ $status -eq 0 ]; then
        log_info "Phase 4 completed successfully for dataset: $dataset"
        grep -A 10 "Formula Summary" "$log_dir/stdout.log" | head -n 10
    else
        log_error "Phase 4 failed with exit code $status for dataset: $dataset"
    fi
    
    return $status
}

# Function to get feature file based on dataset and eval mode
get_feature_file() {
    local dataset="$1"
    local eval_mode="$2"
    local feature_file=""
    
    # Check for custom features set first
    if [ -n "$FEATURES_SET" ]; then
        feature_file="${FEATURE_SETS_DIR}/${FEATURES_SET}.json"
    else
        # Try to use features from corresponding phase3 with parameter suffix
        feature_file="$OUTPUT_DIR/phase3/${dataset}_${eval_mode}_eval${PARAM_SUFFIX}/models/model_features.json"
        
        # Fall back to other options if not found
        if [ ! -f "$feature_file" ]; then
            if [ "$eval_mode" = "none" ]; then
                # Try external or all evaluation mode
                feature_file="$OUTPUT_DIR/phase3/${dataset}_external_eval${PARAM_SUFFIX}/models/model_features.json"
                if [ ! -f "$feature_file" ]; then
                    feature_file="$OUTPUT_DIR/phase3/${dataset}_all_eval${PARAM_SUFFIX}/models/model_features.json"
                fi
            elif [ "$eval_mode" = "external" ]; then
                # Try none or all evaluation mode
                feature_file="$OUTPUT_DIR/phase3/${dataset}_none_eval${PARAM_SUFFIX}/models/model_features.json"
                if [ ! -f "$feature_file" ]; then
                    feature_file="$OUTPUT_DIR/phase3/${dataset}_all_eval${PARAM_SUFFIX}/models/model_features.json"
                fi
            elif [ "$eval_mode" = "all" ]; then
                # Try external or none evaluation mode
                feature_file="$OUTPUT_DIR/phase3/${dataset}_external_eval${PARAM_SUFFIX}/models/model_features.json"
                if [ ! -f "$feature_file" ]; then
                    feature_file="$OUTPUT_DIR/phase3/${dataset}_none_eval${PARAM_SUFFIX}/models/model_features.json"
                fi
            fi
        fi
    fi
    
    # Return empty if not found
    if [ ! -f "$feature_file" ] && [ -n "$feature_file" ]; then
        echo ""
    else
        echo "$feature_file"
    fi
}

# Phase 4 - Claude only
run_phase4_claude_no_eval() {
    local feature_file=$(get_feature_file "claude" "none")
    if [ -n "$feature_file" ]; then
        run_phase4 "claude" "" "none" "$feature_file"
    else
        run_phase4 "claude" "" "none"
    fi
}

run_phase4_claude_external_eval() {
    local external_datasets=$(get_external_datasets "claude")
    local feature_file=$(get_feature_file "claude" "external")
    if [ -n "$feature_file" ]; then
        run_phase4 "claude" "$external_datasets" "external" "$feature_file"
    else
        run_phase4 "claude" "$external_datasets" "external"
    fi
}

run_phase4_claude_all_eval() {
    local feature_file=$(get_feature_file "claude" "all")
    if [ -n "$feature_file" ]; then
        run_phase4 "claude" "$ALL_EVAL_DATASETS" "all" "$feature_file"
    else
        run_phase4 "claude" "$ALL_EVAL_DATASETS" "all"
    fi
}

# Phase 4 - Combined dataset
run_phase4_combined_no_eval() {
    local feature_file=$(get_feature_file "$COMBINED_DATASET" "none")
    if [ -n "$feature_file" ]; then
        run_phase4 "$COMBINED_DATASET" "" "none" "$feature_file"
    else
        run_phase4 "$COMBINED_DATASET" "" "none"
    fi
}

run_phase4_combined_external_eval() {
    local external_datasets=$(get_external_datasets "$COMBINED_DATASET")
    local feature_file=$(get_feature_file "$COMBINED_DATASET" "external")
    if [ -n "$feature_file" ]; then
        run_phase4 "$COMBINED_DATASET" "$external_datasets" "external" "$feature_file"
    else
        run_phase4 "$COMBINED_DATASET" "$external_datasets" "external"
    fi
}

run_phase4_combined_all_eval() {
    local feature_file=$(get_feature_file "$COMBINED_DATASET" "all")
    if [ -n "$feature_file" ]; then
        run_phase4 "$COMBINED_DATASET" "$ALL_EVAL_DATASETS" "all" "$feature_file"
    else
        run_phase4 "$COMBINED_DATASET" "$ALL_EVAL_DATASETS" "all"
    fi
}

#===============================================================================
# 10. MAIN EXECUTION
#===============================================================================

main() {
    setup_environment
    parse_arguments "$@"
    
    log_section "Starting Pipeline - Job ID: $SLURM_JOB_ID"
    
    for set_name in "${SET_LIST[@]}"; do
        case "$set_name" in
            phase1_all)
                run_phase1_all
                ;;
            phase2_all)
                run_phase2_all
                ;;
                
            # Phase 3 - Claude only
            phase3_claude_no_eval)
                run_phase3_claude_no_eval
                ;;
            phase3_claude_external_eval)
                run_phase3_claude_external_eval
                ;;
            phase3_claude_all_eval)
                run_phase3_claude_all_eval
                ;;
                
            # Phase 3 - Combined dataset
            phase3_combined_no_eval)
                run_phase3_combined_no_eval
                ;;
            phase3_combined_external_eval)
                run_phase3_combined_external_eval
                ;;
            phase3_combined_all_eval)
                run_phase3_combined_all_eval
                ;;
                
            # Phase 4 - Claude only
            phase4_claude_no_eval)
                run_phase4_claude_no_eval
                ;;
            phase4_claude_external_eval)
                run_phase4_claude_external_eval
                ;;
            phase4_claude_all_eval)
                run_phase4_claude_all_eval
                ;;
                
            # Phase 4 - Combined dataset
            phase4_combined_no_eval)
                run_phase4_combined_no_eval
                ;;
            phase4_combined_external_eval)
                run_phase4_combined_external_eval
                ;;
            phase4_combined_all_eval)
                run_phase4_combined_all_eval
                ;;
        esac
    done
    
    log_section "Batch Job Completed"
    log_info "All selected configuration sets have been processed."
    log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
    
    deactivate
}

# Execute main function with all script arguments
main "$@"