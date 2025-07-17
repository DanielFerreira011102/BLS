#!/bin/bash
#SBATCH --job-name=bls_metrics    # Name of the job visible in queue
#SBATCH --nodes=1                 # Number of compute nodes to allocate
#SBATCH --ntasks=1                # Number of tasks (processes) to create
#SBATCH --cpus-per-task=12        # CPU cores per task
#SBATCH --mem=40G                 # Memory allocation per node
#SBATCH --gres=gpu:1              # GPU resource allocation (1 GPU)
#SBATCH --partition=gpu           # Compute partition/queue to use
#SBATCH --chdir=/data/home/djbf/storage/bls/rq3  # Working directory
#SBATCH --output=/data/home/djbf/storage/bls/rq3/logs/%j/logs.out  # Standard output file (%j = job ID)
#SBATCH --error=/data/home/djbf/storage/bls/rq3/logs/%j/logs.err   # Standard error file (%j = job ID)

#===============================================================================
# Biomedical Language Simplification (BLS) - Text Readability Analysis
# This script calculates readability metrics for response texts in JSON format
#===============================================================================

#===============================================================================
# GLOBAL VARIABLES
#===============================================================================

# Define base directories
HOME_DIR="/data/home/djbf"
SOURCE_DIR="$HOME_DIR/bls/rq3"
BASE_DIR="$HOME_DIR/storage/bls"
RESOURCE_DIR="$BASE_DIR/resources"
WORK_DIR="$BASE_DIR/rq3"
NLP_DIR="$RESOURCE_DIR/nlp"

# Default values
BATCH_SIZE=32
SPACY_MODEL_SYNTAX="en_core_web_trf"
SPACY_MODEL_UMLS="en_core_sci_scibert"
SPACY_MODEL_JARGON="en_core_web_trf"
COMMONLIT_MODEL_TYPE="albert"
MASKED_PROB_MODEL="emilyalsentzer/Bio_ClinicalBERT"
MASKED_PROB_METHOD="random"
NUM_MASKS=30
MASK_FRACTION=0.15
USE_CORENLP=false
CORENLP_MEMORY="4G"
SCIGISPY_WINDOW_SIZE=1
SCIGISPY_BREAKPOINT_PERCENTILE=75
LLM_MODEL_PATH="$BASE_DIR/rq2/outputs/llmify/default/improved/mse_1.0+kl_0.0+ce_0.0+soft_0.0"
CLUSTER_MODEL_PATH="$RESOURCE_DIR/models/cluster-readability-model-svr-300/medreadme"
JARGON_MODEL_PATH="$RESOURCE_DIR/models/jargon-readability-model-crf/medreadme/best_model"
JARGON_PRETRAINED_MODEL="roberta-large"

# GisPy and SciGisPy specific flags
DISABLE_GISPY_COREF=false
DISABLE_GISPY_PCREF=false
DISABLE_GISPY_PCDC=false
DISABLE_GISPY_CAUSALITY=false
DISABLE_GISPY_CONCRETENESS=false
DISABLE_GISPY_WORDNET=false

DISABLE_SCIGISPY_HYPERNYMY=false
DISABLE_SCIGISPY_VERB_OVERLAP=false
DISABLE_SCIGISPY_COHESION=false
DISABLE_SCIGISPY_IC=false

# Dataset paths (UPDATED to include finetuned3b and claude)
DATASET_PATHS=(
    "baseline:$BASE_DIR/rq3/outputs/phase1/baseline/answer_variants.json"
    "fewshot:$BASE_DIR/rq3/outputs/phase1/fewshot/answer_variants.json"
    "finetuned:$BASE_DIR/rq3/outputs/phase1/finetuned/answer_variants.json"
    "finetuned3b:$BASE_DIR/rq3/outputs/phase1/finetuned3b/answer_variants.json"
    "finetunedIt":"$BASE_DIR/rq3/outputs/phase1/finetunedIt/answer_variants.json"
    "claude:$BASE_DIR/rq3/outputs/phase1/claude/answer_variants.json"
)

# List of all metrics
ALL_METRICS=(textstat syntax umls commonlit masked_prob gispy scigispy cluster llm jargon)

#===============================================================================
# LOGGING FUNCTIONS
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

#===============================================================================
# HELPER FUNCTIONS
#===============================================================================

# Show usage information
show_usage() {
    echo "Usage: sbatch $0 --dataset NAME --metric NAME [OPTIONS]"
    echo ""
    echo "Required Arguments:"
    echo "  --dataset NAME          Dataset to process (baseline, fewshot, finetuned, finetuned3b, finetunedIt, claude, or custom)"
    echo "  --metric NAME           Metric to compute (e.g., textstat, syntax, umls, commonlit, masked_prob, gispy, scigispy, cluster, llm, jargon)"
    echo ""
    echo "Dataset Options:"
    echo "  --custom-dataset-path FILE  Path to custom dataset (required when --dataset=custom)"
    echo "  --output-dir DIR        Base output directory (default: \$WORK_DIR/outputs/phase2)"
    echo ""
    echo "Optional Arguments:"
    echo "  --batch-size SIZE      Batch size for processing (default: 32)"
    echo "  --spacy-model-syntax MODEL spaCy model for syntax metric (default: en_core_web_trf)"
    echo "  --spacy-model-umls MODEL spaCy model for UMLS metric (default: en_core_sci_scibert)"
    echo "  --spacy-model-jargon MODEL spaCy model for jargon metric (default: en_core_web_trf)"
    echo "  --commonlit-model-type TYPE CommonLit model type (default: albert)"
    echo "  --masked-prob-model MODEL Model for masked probability (default: emilyalsentzer/Bio_ClinicalBERT)"
    echo "  --masked-prob-method METHOD Method for masked probability: random, np, or rnptc (default: rnptc)"
    echo "  --num-masks NUM           Number of masks to use for masked_prob metric (default: 10)"
    echo "  --mask-fraction FRAC      Fraction of text to mask for masked_prob metric (default: 0.15)"
    echo "  --use-corenlp           Use CoreNLP for GisPy metric (NOT recommended - much slower)"
    echo "  --corenlp-memory MEM    Memory allocation for CoreNLP (default: 4G)"
    echo ""
    echo "GisPy Specific Options:"
    echo "  --disable-gispy-coref     Disable coreference resolution in GisPy"
    echo "  --disable-gispy-pcref     Disable referential cohesion in GisPy"
    echo "  --disable-gispy-pcdc      Disable deep cohesion in GisPy"
    echo "  --disable-gispy-causality Disable causality metrics in GisPy"
    echo "  --disable-gispy-concreteness Disable concreteness metrics in GisPy"
    echo "  --disable-gispy-wordnet   Disable WordNet-based metrics in GisPy"
    echo ""
    echo "SciGisPy Specific Options:"
    echo "  --disable-scigispy-hypernymy   Disable hypernymy metrics in SciGisPy"
    echo "  --disable-scigispy-verb-overlap Disable verb overlap metrics in SciGisPy"
    echo "  --disable-scigispy-cohesion    Disable cohesion metrics in SciGisPy"
    echo "  --disable-scigispy-ic          Disable information content metrics in SciGisPy"
    echo "  --scigispy-window-size SIZE    Window size for SciGisPy (default: 1)"
    echo "  --scigispy-breakpoint-percentile PERC Breakpoint percentile for SciGisPy (default: 75)"
    echo ""
    echo "Other Specific Options:"
    echo "  --llm-model-path PATH   Path to LLM model (default: $BASE_DIR/rq2/outputs/llmify/default/improved/mse_1.0+kl_0.0+ce_0.0+soft_0.0)"
    echo "  --cluster-model-path PATH Path to cluster model (default: $RESOURCE_DIR/models/cluster-readability-model-svr-300/medreadme)"
    echo "  --jargon-model-path PATH Path to jargon model (default: $RESOURCE_DIR/models/jargon-readability-model-crf/medreadme/best_model)"
    echo "  --jargon-pretrained-model MODEL Pretrained model for jargon classifier (default: roberta-large)"
    echo ""
    echo "Examples:"
    echo "  sbatch $0 --dataset baseline --metric textstat"
    echo "  sbatch $0 --dataset fewshot --metric gispy --disable-gispy-coref"
    echo "  sbatch $0 --dataset claude --metric scigispy --disable-scigispy-ic"
    echo "  sbatch $0 --dataset custom --custom-dataset-path my_data.json --metric textstat"
    exit 1
}

# Setup directories needed for execution
setup_directories() {
    log_info "Creating output directories..."
    
    # Define job-specific directories
    LOGS_DIR="$WORK_DIR/logs/$SLURM_JOB_ID"
    OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/outputs/phase2}"
    
    # Create required directories
    mkdir -p "$OUTPUT_DIR" "$LOGS_DIR"
    check_status "Failed to create output directories"
}

# Setup environment for execution
setup_environment() {
    log_section "Environment Setup"

    # Load required system modules
    log_info "Loading required modules..."
    module load python
    check_status "Failed to load Python module"

    module load gcc/13.2.0-l6taobu
    check_status "Failed to load GCC module"

    module load openjdk
    check_status "Failed to load OpenJDK module"

    module load cuda/12.4.0-x4k27pl
    check_status "Failed to load CUDA module"

    # Set environment variables
    CUDA_LIB_PATH="/apps/local/spack/spack/linux-ubuntu22.04-x86_64_v2/gcc-13.2.0/cuda-12.4.0-x4k27plupvaengzt57xrhk7mijgplgem/lib64"
    export LD_LIBRARY_PATH="$(gcc -print-file-name=libstdc++.so.6 | xargs dirname):$CUDA_LIB_PATH:$LD_LIBRARY_PATH"
    export CUDA_VISIBLE_DEVICES=0
    export NLTK_DATA="$NLP_DIR/WordNetNLTK:$NLP_DIR/PunktNLTK:$NLTK_DATA"

    # Activate virtual environment
    log_info "Activating virtual environment..."
    source "$HOME_DIR/bls/venv/bin/activate"
    check_status "Failed to activate virtual environment"

    # Set CORENLP_HOME only if explicitly using CoreNLP
    if [ "$USE_CORENLP" == "true" ]; then
        export CORENLP_HOME="$NLP_DIR/StanzaCoreNLP"
        log_warn "WARNING: Using CoreNLP for coreference resolution. This is significantly slower."
        log_warn "Consider using the default fastcoref implementation instead."
    fi

    # Verify GPU availability
    log_section "GPU Information"
    nvidia-smi
    check_status "Failed to query GPU information"
}

# Parse command line arguments
parse_arguments() {
    log_section "Argument Parsing"
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset) DATASET="$2"; shift 2 ;;
            --custom-dataset-path) CUSTOM_DATASET_PATH="$2"; shift 2 ;;
            --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
            --metric) METRIC="$2"; shift 2 ;;
            --batch-size) BATCH_SIZE="$2"; shift 2 ;;
            --spacy-model-syntax) SPACY_MODEL_SYNTAX="$2"; shift 2 ;;
            --spacy-model-umls) SPACY_MODEL_UMLS="$2"; shift 2 ;;
            --spacy-model-jargon) SPACY_MODEL_JARGON="$2"; shift 2 ;;
            --commonlit-model-type) COMMONLIT_MODEL_TYPE="$2"; shift 2 ;;
            --masked-prob-model) MASKED_PROB_MODEL="$2"; shift 2 ;;
            --masked-prob-method) MASKED_PROB_METHOD="$2"; shift 2 ;;
            --num-masks) NUM_MASKS="$2"; shift 2 ;;
            --mask-fraction) MASK_FRACTION="$2"; shift 2 ;;
            --use-corenlp) USE_CORENLP=true; shift ;;
            --corenlp-memory) CORENLP_MEMORY="$2"; shift 2 ;;
            --scigispy-window-size) SCIGISPY_WINDOW_SIZE="$2"; shift 2 ;;
            --scigispy-breakpoint-percentile) SCIGISPY_BREAKPOINT_PERCENTILE="$2"; shift 2 ;;
            --llm-model-path) LLM_MODEL_PATH="$2"; shift 2 ;;
            --cluster-model-path) CLUSTER_MODEL_PATH="$2"; shift 2 ;;
            --jargon-model-path) JARGON_MODEL_PATH="$2"; shift 2 ;;
            --jargon-pretrained-model) JARGON_PRETRAINED_MODEL="$2"; shift 2 ;;
            
            # GisPy specific flags
            --disable-gispy-coref) DISABLE_GISPY_COREF=true; shift ;;
            --disable-gispy-pcref) DISABLE_GISPY_PCREF=true; shift ;;
            --disable-gispy-pcdc) DISABLE_GISPY_PCDC=true; shift ;;
            --disable-gispy-causality) DISABLE_GISPY_CAUSALITY=true; shift ;;
            --disable-gispy-concreteness) DISABLE_GISPY_CONCRETENESS=true; shift ;;
            --disable-gispy-wordnet) DISABLE_GISPY_WORDNET=true; shift ;;
            
            # SciGisPy specific flags
            --disable-scigispy-hypernymy) DISABLE_SCIGISPY_HYPERNYMY=true; shift ;;
            --disable-scigispy-verb-overlap) DISABLE_SCIGISPY_VERB_OVERLAP=true; shift ;;
            --disable-scigispy-cohesion) DISABLE_SCIGISPY_COHESION=true; shift ;;
            --disable-scigispy-ic) DISABLE_SCIGISPY_IC=true; shift ;;
            
            --help|-h) show_usage ;;
            *) log_error "Unknown argument: $1"; show_usage ;;
        esac
    done

    # Validate required arguments
    if [ -z "$DATASET" ] || [ -z "$METRIC" ]; then
        log_error "Required arguments: --dataset and --metric"
        show_usage
    fi
}

# Validate selected dataset and input file
validate_dataset() {
    # Validate dataset selection
    case "$DATASET" in
        baseline|fewshot|finetuned|finetuned3b|finetunedIt|claude)
            # Find dataset path
            for dataset_path in "${DATASET_PATHS[@]}"; do
                IFS=':' read -r ds_name ds_path <<< "$dataset_path"
                if [ "$ds_name" == "$DATASET" ]; then
                    INPUT_FILE="$ds_path"
                    break
                fi
            done
            ;;
        custom)
            if [ -z "$CUSTOM_DATASET_PATH" ]; then
                log_error "Custom dataset requires --custom-dataset-path"
                show_usage
            fi
            INPUT_FILE="$CUSTOM_DATASET_PATH"
            ;;
        *)
            log_error "Invalid dataset: $DATASET. Choose from baseline, fewshot, finetuned, finetuned3b, claude, or custom."
            show_usage
            ;;
    esac

    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        log_error "Input file does not exist: $INPUT_FILE"
        exit 1
    fi
    
    log_info "Dataset: $DATASET"
    log_info "Input file: $INPUT_FILE"
}

# Configure metric-specific settings and output paths
configure_metric() {
    log_section "Metric Configuration"

    # Generate disable flags for all metrics except the selected one
    DISABLE_FLAGS=""
    for m in "${ALL_METRICS[@]}"; do
        if [ "$m" != "$METRIC" ]; then
            DISABLE_FLAGS+=" --disable-${m//_/-}"
        fi
    done

    # Configure metric-specific settings and output subdirectories
    case "$METRIC" in
        textstat)
            SUBDIR="textstat"
            EXTRA_FLAGS=""
            ;;
        syntax)
            SUBDIR="syntax/$SPACY_MODEL_SYNTAX"
            EXTRA_FLAGS="--spacy-model-syntax $SPACY_MODEL_SYNTAX"
            ;;
        umls)
            SUBDIR="umls/$SPACY_MODEL_UMLS"
            EXTRA_FLAGS="--spacy-model-umls $SPACY_MODEL_UMLS"
            ;;
        commonlit)
            SUBDIR="commonlit/$COMMONLIT_MODEL_TYPE"
            EXTRA_FLAGS="--commonlit-model-type $COMMONLIT_MODEL_TYPE"
            ;;
        masked_prob)
            MASKED_PROB_MODEL_NAME=$(basename "$MASKED_PROB_MODEL")
            SUBDIR="masked_prob/$MASKED_PROB_METHOD/$MASKED_PROB_MODEL_NAME"
            EXTRA_FLAGS="--masked-prob-model $MASKED_PROB_MODEL --masked-prob-method $MASKED_PROB_METHOD --num-masks $NUM_MASKS --mask-fraction $MASK_FRACTION"
            ;;
        gispy)
            SUBDIR="gispy"
            EXTRA_FLAGS=""
            
            # Add specific GisPy flags
            if [ "$USE_CORENLP" == "true" ]; then
                SUBDIR="gispy/corenlp"
                EXTRA_FLAGS+=" --use-corenlp --corenlp-memory $CORENLP_MEMORY"
            fi
            
            # Add disabled GisPy sub-metrics
            if [ "$DISABLE_GISPY_COREF" == "true" ]; then
                EXTRA_FLAGS+=" --disable-gispy-coref"
            fi
            if [ "$DISABLE_GISPY_PCREF" == "true" ]; then
                EXTRA_FLAGS+=" --disable-gispy-pcref"
            fi
            if [ "$DISABLE_GISPY_PCDC" == "true" ]; then
                EXTRA_FLAGS+=" --disable-gispy-pcdc"
            fi
            if [ "$DISABLE_GISPY_CAUSALITY" == "true" ]; then
                EXTRA_FLAGS+=" --disable-gispy-causality"
            fi
            if [ "$DISABLE_GISPY_CONCRETENESS" == "true" ]; then
                EXTRA_FLAGS+=" --disable-gispy-concreteness"
            fi
            if [ "$DISABLE_GISPY_WORDNET" == "true" ]; then
                EXTRA_FLAGS+=" --disable-gispy-wordnet"
            fi
            ;;
        scigispy)
            if [ "$SCIGISPY_WINDOW_SIZE" != "1" ] || [ "$SCIGISPY_BREAKPOINT_PERCENTILE" != "75" ]; then
                SUBDIR="scigispy/ws${SCIGISPY_WINDOW_SIZE}_bp${SCIGISPY_BREAKPOINT_PERCENTILE}"
            else
                SUBDIR="scigispy"
            fi
            EXTRA_FLAGS="--scigispy-window-size $SCIGISPY_WINDOW_SIZE --scigispy-breakpoint-percentile $SCIGISPY_BREAKPOINT_PERCENTILE"
            
            # Add disabled SciGisPy sub-metrics
            if [ "$DISABLE_SCIGISPY_HYPERNYMY" == "true" ]; then
                EXTRA_FLAGS+=" --disable-scigispy-hypernymy"
            fi
            if [ "$DISABLE_SCIGISPY_VERB_OVERLAP" == "true" ]; then
                EXTRA_FLAGS+=" --disable-scigispy-verb-overlap"
            fi
            if [ "$DISABLE_SCIGISPY_COHESION" == "true" ]; then
                EXTRA_FLAGS+=" --disable-scigispy-cohesion"
            fi
            if [ "$DISABLE_SCIGISPY_IC" == "true" ]; then
                EXTRA_FLAGS+=" --disable-scigispy-ic"
            fi
            ;;
        cluster)
            CLUSTER_MODEL_NAME=$(basename "$CLUSTER_MODEL_PATH")
            SUBDIR="cluster/$CLUSTER_MODEL_NAME"
            EXTRA_FLAGS="--cluster-model-path $CLUSTER_MODEL_PATH"
            ;;
        llm)
            LLM_MODEL_NAME=$(basename "$LLM_MODEL_PATH")
            SUBDIR="llm/$LLM_MODEL_NAME"
            EXTRA_FLAGS="--llm-model-path $LLM_MODEL_PATH"
            ;;
        jargon)
            JARGON_PRETRAINED_MODEL_NAME=$(basename "$JARGON_PRETRAINED_MODEL")
            SUBDIR="jargon/$JARGON_PRETRAINED_MODEL_NAME"
            EXTRA_FLAGS="--jargon-model-path $JARGON_MODEL_PATH --jargon-pretrained-model $JARGON_PRETRAINED_MODEL --spacy-model-jargon $SPACY_MODEL_JARGON"
            ;;
        *)
            log_error "Unknown metric: $METRIC"
            exit 1
            ;;
    esac

    # Define output directory and file
    OUTPUT_SUBDIR="$OUTPUT_DIR/$DATASET/$SUBDIR"
    OUTPUT_FILE="$OUTPUT_SUBDIR/readability_metrics.json"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_SUBDIR"
    check_status "Failed to create output directory: $OUTPUT_SUBDIR"

    log_info "Output directory: $OUTPUT_SUBDIR"
    log_info "Output file: $OUTPUT_FILE"
    log_info "Selected metric: $METRIC"
}

# Execute the main Python script with constructed command
execute_script() {
    log_section "Starting Job Execution"

    # Path to the readability scorer script
    SCRIPT_PATH="$SOURCE_DIR/annotate/phase2.py"

    # Construct the Python command with parameters
    PYTHON_CMD="python $SCRIPT_PATH --dataset $DATASET"

    # Add custom dataset path if used
    if [ "$DATASET" == "custom" ]; then
        PYTHON_CMD+=" --custom-dataset-path $CUSTOM_DATASET_PATH"
    fi

    # Add output file and remaining parameters
    PYTHON_CMD+=" --output $OUTPUT_FILE $DISABLE_FLAGS $EXTRA_FLAGS --batch-size $BATCH_SIZE"

    # Add explicit garbage collection
    log_info "Adding explicit garbage collection module"
    PYTHON_CMD="python -c \"import gc; gc.enable();\" && $PYTHON_CMD"

    # Run the command
    log_info "Executing: $PYTHON_CMD"
    eval "$PYTHON_CMD"
    execution_status=$?

    if [ $execution_status -eq 0 ]; then
        log_section "Job Completed Successfully"
        log_info "Readability metrics for $DATASET saved to: $OUTPUT_FILE"
    else
        log_section "Job Failed"
        log_error "Python script exited with status code: $execution_status"
        log_error "Check logs for more details: $LOGS_DIR/logs.err"
        exit $execution_status
    fi
}

# Cleanup and finalize execution
cleanup() {
    # Deactivate the virtual environment
    deactivate

    log_info "Job ID: $SLURM_JOB_ID completed at $(date)"
}

#===============================================================================
# MAIN SCRIPT EXECUTION
#===============================================================================

main() {
    # Parse all command line arguments
    parse_arguments "$@"
    
    # Setup required directories
    setup_directories
    
    # Setup execution environment
    setup_environment
    
    # Validate dataset selection
    validate_dataset
    
    # Configure metric settings
    configure_metric
    
    # Execute the script
    execute_script
    
    # Cleanup after execution
    cleanup
}

# Run the main function with all script arguments
main "$@"

#===============================================================================
# USAGE EXAMPLES (For reference only, not executed)
#===============================================================================

# Example 1: Basic usage with TextStat metrics
#   sbatch run.sh --dataset baseline --metric textstat

# Example 2: Using new finetuned3b dataset with syntax metrics
#   sbatch run.sh --dataset finetuned3b --metric syntax --spacy-model-syntax en_core_web_trf

# Example 3: Using new claude dataset with GisPy metrics
#   sbatch run.sh --dataset claude --metric gispy --disable-gispy-pcref

# Example 4: SciGisPy with specific sub-metrics disabled and custom parameters
#   sbatch run.sh --dataset baseline --metric scigispy --disable-scigispy-hypernymy --disable-scigispy-verb-overlap

# Example 5: Cluster metrics with specific model
#   sbatch run.sh --dataset finetuned3b --metric cluster

# Example 6: LLM metrics with specific model on the claude dataset
#   sbatch run.sh --dataset claude --metric llm

# Example 7: Albert model for CommonLit metrics
#   sbatch run.sh --dataset fewshot --metric commonlit --commonlit-model-type albert

# Example 8: Masked probability metrics with specific model and method
#   sbatch run.sh --dataset finetuned --metric masked_prob --masked-prob-model emilyalsentzer/Bio_ClinicalBERT