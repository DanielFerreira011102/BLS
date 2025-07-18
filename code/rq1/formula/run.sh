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
BOOTSTRAP_SAMPLES=1000
FINAL_FEATURE_COUNT=""  # Default empty, will use all features
CORR_THRESHOLD=0.7

# Define valid sets
VALID_SETS=(
    "all"
    "phase1_all"
    "phase2_all"
    
    # Phase 3 - Simplified feature selection
    "phase3_claude"
    "phase3_combined"
    
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
                   
                   # Phase 3 - Feature selection
                   phase3_claude                - Run Phase 3 for Claude dataset
                   phase3_combined              - Run Phase 3 for combined dataset
                   
                   # Phase 4 - Claude only
                   phase4_claude_no_eval        - Run Phase 4 for Claude without evaluation
                   phase4_claude_external_eval  - Run Phase 4 for Claude with external evaluation
                   phase4_claude_all_eval       - Run Phase 4 for Claude with all evaluation
                   
                   # Phase 4 - Combined dataset
                   phase4_combined_no_eval      - Run Phase 4 for combined without evaluation
                   phase4_combined_external_eval - Run Phase 4 for combined with external evaluation
                   phase4_combined_all_eval     - Run Phase 4 for combined with all evaluation
                   
                 Default: all
  --features SET             Specify a feature set to use for Phase 4 (e.g., core7, lasso13)
                             Feature set must exist in $FEATURE_SETS_DIR/
  --bootstrap-samples N      Number of bootstrap samples for Phase 3 (default: 1000)
  --final-feature-count N    Number of features to retain in Phase 3 (default: None, all features)
  --corr-threshold VALUE     Correlation threshold for Phase 3 (default: 0.7)
  --help|-h                  Show this help message
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
            --bootstrap-samples) BOOTSTRAP_SAMPLES="$2"; shift 2 ;;
            --final-feature-count) FINAL_FEATURE_COUNT="$2"; shift 2 ;;
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
    
    # Log configuration
    log_info "Sets to run: ${SET_LIST[*]}"
    if [ -n "$FEATURES_SET" ]; then
        log_info "Features set: $FEATURES_SET"
    fi
    log_info "Bootstrap samples: $BOOTSTRAP_SAMPLES"
    log_info "Final feature count: $FINAL_FEATURE_COUNT"
    log_info "Correlation threshold: $CORR_THRESHOLD"
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
    local log_dir="$LOGS_DIR/phase3_${dataset}"
    
    log_section "Running Phase 3 for dataset: $dataset"
    
    mkdir -p "$log_dir"
    
    local phase1_dir="$OUTPUT_DIR/phase1"
    local output_dir="$OUTPUT_DIR/phase3/${dataset}"
    
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
    
    # Add phase 3 specific options
    cmd="$cmd --bootstrap-samples $BOOTSTRAP_SAMPLES"
    
    # If final feature count is specified, add it
    if [ -n "$FINAL_FEATURE_COUNT" ]; then
        cmd="$cmd --final-feature-count $FINAL_FEATURE_COUNT"
    else
        log_info "No final feature count specified, using all features"
    fi
    
    cmd="$cmd --correlation-threshold $CORR_THRESHOLD"
    
    log_info "Executing: $cmd"
    eval "$cmd" > "$log_dir/stdout.log" 2> "$log_dir/stderr.log"
    
    local status=$?
    if [ $status -eq 0 ]; then
        log_info "Phase 3 completed successfully for dataset: $dataset"
        # Try to show selected features if available
        if [ -f "$output_dir/data/selected_features.csv" ]; then
            log_info "Selected features:"
            head -n 15 "$output_dir/data/selected_features.csv"
        fi
    else
        log_error "Phase 3 failed with exit code $status for dataset: $dataset"
    fi
    
    return $status
}

# Phase 3 functions
run_phase3_claude() {
    run_phase3 "claude"
}

run_phase3_combined() {
    run_phase3 "$COMBINED_DATASET"
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
    
    # Create directory name
    local dir_suffix="${dataset}_${eval_mode}_eval"
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

# Function to get feature file based on dataset
get_feature_file() {
    local dataset="$1"
    local feature_file=""
    
    # Check for custom features set first
    if [ -n "$FEATURES_SET" ]; then
        feature_file="${FEATURE_SETS_DIR}/${FEATURES_SET}.json"
    else
        # Try to use features from corresponding phase3
        feature_file="$OUTPUT_DIR/phase3/${dataset}/data/selected_features.csv"
        
        # Convert CSV to expected JSON format for phase4 if it exists
        if [ -f "$feature_file" ]; then
            local json_file="$OUTPUT_DIR/phase3/${dataset}/data/selected_features.json"
            if [ ! -f "$json_file" ]; then
                # Create a simple JSON file from CSV for phase4 compatibility
                python3 -c "
import pandas as pd
import json
df = pd.read_csv('$feature_file')
features = df['feature'].tolist()
with open('$json_file', 'w') as f:
    json.dump(features, f)
"
            fi
            feature_file="$json_file"
        else
            feature_file=""
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
    local feature_file=$(get_feature_file "claude")
    if [ -n "$feature_file" ]; then
        run_phase4 "claude" "" "none" "$feature_file"
    else
        run_phase4 "claude" "" "none"
    fi
}

run_phase4_claude_external_eval() {
    local external_datasets=$(get_external_datasets "claude")
    local feature_file=$(get_feature_file "claude")
    if [ -n "$feature_file" ]; then
        run_phase4 "claude" "$external_datasets" "external" "$feature_file"
    else
        run_phase4 "claude" "$external_datasets" "external"
    fi
}

run_phase4_claude_all_eval() {
    local feature_file=$(get_feature_file "claude")
    if [ -n "$feature_file" ]; then
        run_phase4 "claude" "$ALL_EVAL_DATASETS" "all" "$feature_file"
    else
        run_phase4 "claude" "$ALL_EVAL_DATASETS" "all"
    fi
}

# Phase 4 - Combined dataset
run_phase4_combined_no_eval() {
    local feature_file=$(get_feature_file "$COMBINED_DATASET")
    if [ -n "$feature_file" ]; then
        run_phase4 "$COMBINED_DATASET" "" "none" "$feature_file"
    else
        run_phase4 "$COMBINED_DATASET" "" "none"
    fi
}

run_phase4_combined_external_eval() {
    local external_datasets=$(get_external_datasets "$COMBINED_DATASET")
    local feature_file=$(get_feature_file "$COMBINED_DATASET")
    if [ -n "$feature_file" ]; then
        run_phase4 "$COMBINED_DATASET" "$external_datasets" "external" "$feature_file"
    else
        run_phase4 "$COMBINED_DATASET" "$external_datasets" "external"
    fi
}

run_phase4_combined_all_eval() {
    local feature_file=$(get_feature_file "$COMBINED_DATASET")
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
                
            # Phase 3 - Feature selection
            phase3_claude)
                run_phase3_claude
                ;;
            phase3_combined)
                run_phase3_combined
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