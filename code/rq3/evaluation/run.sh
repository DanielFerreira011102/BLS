#!/bin/bash
#SBATCH --job-name=complexity_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --chdir=/data/home/djbf/storage/bls/rq3
#SBATCH --output=/data/home/djbf/storage/bls/rq3/logs/%j/eval_logs.out
#SBATCH --error=/data/home/djbf/storage/bls/rq3/logs/%j/eval_logs.err
#SBATCH --time=12:00:00

#===============================================================================
# Complexity Evaluation - RQ3
#===============================================================================

# Global configuration
readonly HOME_DIR="/data/home/djbf"
readonly SOURCE_DIR="$HOME_DIR/bls/rq3"
readonly BASE_DIR="$HOME_DIR/storage/bls"
readonly WORK_DIR="$BASE_DIR/rq3"
readonly EVAL_SCRIPT="$HOME_DIR/bls/rq3/evaluation/phase1.py"
readonly DATA_PATH="/data/home/djbf/storage/bls/resources/datasets/claude/health_qa_20250222_205257.json"

# Model paths
readonly MODEL_PATH="/data/home/djbf/storage/bls/rq3/outputs/phase0/models/complexity_20250502_055712/final_model"
readonly MODEL_PATH_3B="/data/home/djbf/storage/bls/rq3/outputs/phase0/models/complexity_20250502_054415/final_model"
readonly MODEL_PATH_IT="/data/home/djbf/storage/bls/rq3/outputs/phase0/models/complexity_20250513_221225/final_model"
readonly BASE_MODEL="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
readonly BASE_MODEL_3B="unsloth/Llama-3.2-3B-Instruct-bnb-4bit"

# Default parameters
NUM_QUESTIONS=100
NUM_EXAMPLES=3
COMPLEXITY_MIN=0
COMPLEXITY_MAX=100
COMPLEXITY_STEP=5
EXAMPLES_PATH=""
CLAUDE_API_KEY="your_claude_api_key_here"

# Evaluation flags
EVAL_BASELINE=false
EVAL_FINETUNED=false
EVAL_FINETUNED_3B=false
EVAL_FINETUNED_IT=false
EVAL_FEWSHOT=false
EVAL_CLAUDE=false

# Runtime directories
OUTPUT_DIR=""
LOGS_DIR=""

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
        return 1
    fi
}

#===============================================================================
# UTILITY FUNCTIONS
#===============================================================================

show_usage() {
    cat << EOF
Usage: sbatch $0 [OPTIONS]

Options:
  --baseline              Evaluate baseline model
  --finetuned             Evaluate fine-tuned 8B model
  --finetuned-3b          Evaluate fine-tuned 3B model
  --finetuned-it          Evaluate fine-tuned IT model
  --fewshot               Evaluate few-shot learning approach
  --claude                Evaluate using Claude 3.7 API
  --num-questions N       Number of questions to evaluate (default: all)
  --num-examples N        Number of examples for few-shot (default: 3)
  --examples-path PATH    Path to few-shot examples data (JSON)
  --complexity-min N      Minimum complexity level (default: 0)
  --complexity-max N      Maximum complexity level (default: 100)
  --complexity-step N     Step size for complexity levels (default: 5)
EOF
}

set_default_evaluations() {
    # If no evaluation method specified, run all except Claude (costs money)
    local any_eval_set=false
    
    [[ "$EVAL_BASELINE" == true ]] && any_eval_set=true
    [[ "$EVAL_FINETUNED" == true ]] && any_eval_set=true
    [[ "$EVAL_FINETUNED_3B" == true ]] && any_eval_set=true
    [[ "$EVAL_FINETUNED_IT" == true ]] && any_eval_set=true
    [[ "$EVAL_FEWSHOT" == true ]] && any_eval_set=true
    [[ "$EVAL_CLAUDE" == true ]] && any_eval_set=true
    
    if [[ "$any_eval_set" == false ]]; then
        log_info "No evaluation methods specified. Running baseline, finetuned, and few-shot methods."
        EVAL_BASELINE=true
        EVAL_FINETUNED=true
        EVAL_FEWSHOT=true
    fi
}

log_evaluation_plan() {
    [[ "$EVAL_BASELINE" == true ]] && log_info "Will evaluate baseline model"
    [[ "$EVAL_FINETUNED" == true ]] && log_info "Will evaluate fine-tuned model (8B)"
    [[ "$EVAL_FINETUNED_3B" == true ]] && log_info "Will evaluate fine-tuned 3B model"
    [[ "$EVAL_FINETUNED_IT" == true ]] && log_info "Will evaluate fine-tuned IT model"
    [[ "$EVAL_CLAUDE" == true ]] && log_info "Will evaluate using Claude 3.7 API"
    
    if [[ "$EVAL_FEWSHOT" == true ]]; then
        log_info "Will evaluate few-shot model with $NUM_EXAMPLES examples"
        [[ -n "$EXAMPLES_PATH" ]] && log_info "Using custom examples from: $EXAMPLES_PATH" || log_info "Using default examples"
    fi
}

#===============================================================================
# SETUP FUNCTIONS
#===============================================================================

parse_arguments() {
    log_section "Parsing Arguments"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --baseline) EVAL_BASELINE=true; shift ;;
            --finetuned) EVAL_FINETUNED=true; shift ;;
            --finetuned-3b) EVAL_FINETUNED_3B=true; shift ;;
            --finetuned-it) EVAL_FINETUNED_IT=true; shift ;;
            --fewshot) EVAL_FEWSHOT=true; shift ;;
            --claude) EVAL_CLAUDE=true; shift ;;
            --num-questions) NUM_QUESTIONS="$2"; shift 2 ;;
            --num-examples) NUM_EXAMPLES="$2"; shift 2 ;;
            --examples-path) EXAMPLES_PATH="$2"; shift 2 ;;
            --complexity-min) COMPLEXITY_MIN="$2"; shift 2 ;;
            --complexity-max) COMPLEXITY_MAX="$2"; shift 2 ;;
            --complexity-step) COMPLEXITY_STEP="$2"; shift 2 ;;
            *) log_error "Unknown option: $1"; show_usage; exit 1 ;;
        esac
    done
    
    set_default_evaluations
    log_evaluation_plan
}

setup_directories() {
    LOGS_DIR="$WORK_DIR/logs/$SLURM_JOB_ID"
    OUTPUT_DIR="$WORK_DIR/outputs/evaluation_results"
    
    mkdir -p "$LOGS_DIR" "$OUTPUT_DIR"
    check_status "Failed to create output directories"
}

setup_modules() {
    log_info "Loading required modules..."
    
    module load python
    check_status "Failed to load Python module"
    
    module load gcc/13.2.0-l6taobu
    check_status "Failed to load GCC module"
    
    module load cuda/12.4.0-x4k27pl
    check_status "Failed to load CUDA module"
}

setup_environment_variables() {
    local cuda_lib_path="/apps/local/spack/spack/linux-ubuntu22.04-x86_64_v2/gcc-13.2.0/cuda-12.4.0-x4k27plupvaengzt57xrhk7mijgplgem/lib64"
    
    export LD_LIBRARY_PATH="$(gcc -print-file-name=libstdc++.so.6 | xargs dirname):$cuda_lib_path:$LD_LIBRARY_PATH"
    export CUDA_VISIBLE_DEVICES=0
    export WANDB_API_KEY="your_wandb_api_key_here"
    export HUGGINGFACE_TOKEN="your_huggingface_token_here"
    export ANTHROPIC_API_KEY="$CLAUDE_API_KEY"
}

setup_environment() {
    log_section "Environment Setup"
    
    setup_directories
    setup_modules
    setup_environment_variables
    
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    check_status "Failed to activate virtual environment"
    
    [[ ! -f "$EVAL_SCRIPT" ]] && { log_error "Evaluation script not found: $EVAL_SCRIPT"; exit 1; }
    
    log_section "GPU Information"
    nvidia-smi
    check_status "Failed to query GPU information"
}

#===============================================================================
# EVALUATION FUNCTIONS
#===============================================================================

build_base_command() {
    local output_dir="$1"
    local cmd="python $EVAL_SCRIPT --data-path $DATA_PATH --output-dir $output_dir"
    cmd="$cmd --complexity-min $COMPLEXITY_MIN --complexity-max $COMPLEXITY_MAX --complexity-step $COMPLEXITY_STEP"
    
    [[ -n "$NUM_QUESTIONS" ]] && cmd="$cmd --num-questions $NUM_QUESTIONS"
    
    echo "$cmd"
}

execute_evaluation() {
    local cmd="$1"
    local description="$2"
    
    log_info "Executing: $cmd"
    eval "$cmd"
    check_status "$description failed"
}

run_baseline_evaluation() {
    [[ "$EVAL_BASELINE" != true ]] && return 0
    
    log_section "Baseline Model Evaluation"
    
    local cmd="$(build_base_command "$OUTPUT_DIR") --base-model $BASE_MODEL --eval-baseline"
    execute_evaluation "$cmd" "Baseline model evaluation"
}

run_finetuned_evaluation() {
    [[ "$EVAL_FINETUNED" != true ]] && return 0
    
    log_section "Fine-tuned Model Evaluation (8B)"
    
    local cmd="$(build_base_command "$OUTPUT_DIR") --base-model $BASE_MODEL --ft-model $MODEL_PATH --eval-finetuned"
    execute_evaluation "$cmd" "Fine-tuned model evaluation"
}

run_finetuned_3b_evaluation() {
    [[ "$EVAL_FINETUNED_3B" != true ]] && return 0
    
    log_section "Fine-tuned 3B Model Evaluation"
    
    local output_dir_3b="$OUTPUT_DIR/finetuned_3b"
    mkdir -p "$output_dir_3b"
    check_status "Failed to create 3B output directory"
    
    local cmd="$(build_base_command "$output_dir_3b") --base-model $BASE_MODEL_3B --ft-model $MODEL_PATH_3B --eval-finetuned"
    execute_evaluation "$cmd" "Fine-tuned 3B model evaluation"
    
    log_info "Processing 3B model results"
    rename_and_copy_results "$output_dir_3b" "3b"
}

run_finetuned_it_evaluation() {
    [[ "$EVAL_FINETUNED_IT" != true ]] && return 0
    
    log_section "Fine-tuned IT Model Evaluation"
    
    local output_dir_it="$OUTPUT_DIR/finetuned_it"
    mkdir -p "$output_dir_it"
    check_status "Failed to create IT output directory"
    
    local cmd="$(build_base_command "$output_dir_it") --base-model $BASE_MODEL --ft-model $MODEL_PATH_IT --eval-finetuned --no-control-codes"
    execute_evaluation "$cmd" "Fine-tuned IT model evaluation"
    
    log_info "Processing IT model results"
    rename_and_copy_results "$output_dir_it" "it"
}

run_fewshot_evaluation() {
    [[ "$EVAL_FEWSHOT" != true ]] && return 0
    
    log_section "Few-shot Evaluation"
    
    local cmd="$(build_base_command "$OUTPUT_DIR") --base-model $BASE_MODEL --eval-fewshot --num-examples $NUM_EXAMPLES"
    [[ -n "$EXAMPLES_PATH" ]] && cmd="$cmd --examples-path $EXAMPLES_PATH"
    
    execute_evaluation "$cmd" "Few-shot evaluation"
}

run_claude_evaluation() {
    [[ "$EVAL_CLAUDE" != true ]] && return 0
    
    log_section "Claude 3.7 API Evaluation"
    
    local cmd="$(build_base_command "$OUTPUT_DIR") --base-model $BASE_MODEL --eval-claude --claude-api-key $CLAUDE_API_KEY"
    execute_evaluation "$cmd" "Claude API evaluation"
}

rename_and_copy_results() {
    local source_dir="$1"
    local model_suffix="$2"
    
    # Rename result files
    mv "$source_dir/finetuned_results.json" "$source_dir/finetuned_${model_suffix}_results.json" 2>/dev/null
    mv "$source_dir/all_results.json" "$source_dir/all_${model_suffix}_results.json" 2>/dev/null
    mv "$source_dir/all_results.csv" "$source_dir/all_${model_suffix}_results.csv" 2>/dev/null
    
    # Copy results to main output directory
    cp "$source_dir/finetuned_${model_suffix}_results.json" "$OUTPUT_DIR/" 2>/dev/null
    cp "$source_dir/all_${model_suffix}_results.json" "$OUTPUT_DIR/" 2>/dev/null
    cp "$source_dir/all_${model_suffix}_results.csv" "$OUTPUT_DIR/" 2>/dev/null
}

#===============================================================================
# MAIN EXECUTION
#===============================================================================

run_all_evaluations() {
    log_section "Starting Complexity Evaluation - Job ID: $SLURM_JOB_ID"
    
    run_baseline_evaluation
    run_finetuned_evaluation
    run_finetuned_3b_evaluation
    run_finetuned_it_evaluation
    run_fewshot_evaluation
    run_claude_evaluation
}

main() {
    parse_arguments "$@"
    setup_environment
    run_all_evaluations
    
    log_section "Batch Job Completed"
    log_info "Complexity evaluation processed."
    log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
    log_info "Results saved to: $OUTPUT_DIR"
    
    deactivate
}

main "$@"