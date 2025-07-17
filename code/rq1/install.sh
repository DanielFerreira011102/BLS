#!/bin/bash
#===============================================================================
# Biomedical Language Simplification (BLS) Environment Setup
# This script installs all dependencies for the BLS project
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

#===============================================================================
# CONFIGURATION
#===============================================================================

log_section "Configuration"

# Define paths and versions
VENV_PATH="../venv"
BLS_RESOURCES="$HOME/storage/bls/resources"
UMLS_PATH="$BLS_RESOURCES/umls-2024AB-metathesaurus-full/2024AB/META"
QUICKUMLS_PATH="$BLS_RESOURCES/QuickUMLS"
SCISPACY_VERSION="0.5.4"
SCISPACY_MODEL="en_core_sci_sm"

log_info "Virtual environment path: $VENV_PATH"
log_info "BLS resources directory: $BLS_RESOURCES"
log_info "UMLS Metathesaurus path: $UMLS_PATH"
log_info "QuickUMLS installation path: $QUICKUMLS_PATH"
log_info "SciSpaCy version: $SCISPACY_VERSION"
log_info "SciSpaCy model: $SCISPACY_MODEL"

#===============================================================================
# VIRTUAL ENVIRONMENT SETUP
#===============================================================================

log_section "Virtual Environment Setup"

# Check if virtual environment exists
if [ -d "$VENV_PATH" ]; then
    log_warn "Existing virtual environment found at $VENV_PATH"
    read -p "Remove existing environment? [y/N] " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_PATH"
    else
        log_info "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Load required modules
log_info "Loading system modules..."
module load python
check_status "Failed to load Python module"

module load gcc/13.2.0-l6taobu
check_status "Failed to load GCC module"

module load openjdk
check_status "Failed to load OpenJDK module"

# Create new virtual environment
log_info "Creating new virtual environment at $VENV_PATH..."
python -m venv "$VENV_PATH"
check_status "Failed to create virtual environment"

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
check_status "Failed to activate virtual environment"

#===============================================================================
# PACKAGE INSTALLATION
#===============================================================================

log_section "Package Installation"

# Install with uv using requirements file
if command -v uv &> /dev/null; then
    log_info "Using uv for faster package installation..."
    uv pip install -r requirements.txt
    check_status "Failed to install packages with uv"
else
    log_warn "uv not found, falling back to standard pip"
    pip install -r requirements.txt
    check_status "Failed to install packages with pip"
fi

#===============================================================================
# SPACY MODELS INSTALLATION
#===============================================================================

log_section "SpaCy Models Installation"

# Install standard spaCy models
log_info "Installing en_core_web_sm model..."
python -m spacy download en_core_web_sm
check_status "Failed to install en_core_web_sm"

log_info "Installing en_core_web_trf model..."
python -m spacy download en_core_web_trf
check_status "Failed to install en_core_web_trf"

#===============================================================================
# SCISPACY MODEL INSTALLATION
#===============================================================================

log_section "SciSpaCy Model Installation"

# Download SciSpaCy model
SCISPACY_FILE="${SCISPACY_MODEL}-${SCISPACY_VERSION}.tar.gz"
if [ ! -f "$SCISPACY_FILE" ]; then
    log_info "Downloading $SCISPACY_FILE..."
    wget "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v${SCISPACY_VERSION}/${SCISPACY_FILE}"
    check_status "Failed to download SciSpaCy model"
else
    log_info "Using existing $SCISPACY_FILE"
fi

# Install SciSpaCy model
log_info "Installing SciSpaCy model..."
pip install "$SCISPACY_FILE"
check_status "Failed to install SciSpaCy model"

#===============================================================================
# QUICKUMLS INSTALLATION
#===============================================================================

log_section "QuickUMLS Installation"

# Check if UMLS path exists
if [ ! -d "$UMLS_PATH" ]; then
    log_error "UMLS path not found: $UMLS_PATH"
    log_error "Please ensure the UMLS Metathesaurus is downloaded and extracted"
    exit 1
fi

# Install QuickUMLS
log_info "Installing QuickUMLS from UMLS data..."
python -m quickumls.install "$UMLS_PATH" "$QUICKUMLS_PATH"
check_status "Failed to install QuickUMLS"

#===============================================================================
# OPTIONAL COMPONENTS
#===============================================================================

log_section "Optional Components (Commented Out)"

# Uncomment and run these commands manually if needed

# Clean up UMLS files to save space
# log_info "Cleaning up UMLS files..."
# rm -rf "$UMLS_PATH"

# Install Stanza CoreNLP
# log_info "Installing Stanza CoreNLP..."
# python -c "import stanza; stanza.install_corenlp(dir='$BLS_RESOURCES/nlp/StanzaCoreNLP')"

# Install NLTK data
# log_info "Installing NLTK data..."
# python -c "import nltk; nltk.download('wordnet', download_dir='$BLS_RESOURCES/nlp/WordNetNLTK'); nltk.download('punkt', download_dir='$BLS_RESOURCES/nlp/PunktNLTK')"

#===============================================================================
# PACKAGE CONFLICTS
#===============================================================================

# Requirement already satisfied: certifi>=2017.4.17 in /data/home/djbf/bls/venv/lib/python3.11/site-packages (from requests->transformers>=4.25->bleurt-pytorch==0.0.1) (2025.1.31)
# Requirement already satisfied: mpmath<1.4,>=1.1.0 in /data/home/djbf/bls/venv/lib/python3.11/site-packages (from sympy->torch>=1.10->bleurt-pytorch==0.0.1) (1.3.0)
# Using cached huggingface_hub-0.30.2-py3-none-any.whl (481 kB)
# Building wheels for collected packages: bleurt-pytorch
#   Building wheel for bleurt-pytorch (pyproject.toml) ... done
#   Created wheel for bleurt-pytorch: filename=bleurt_pytorch-0.0.1-py3-none-any.whl size=22370 sha256=59df7b6abeeeb7b8ba3bc9ca0d623cc354a787e5e437c9556fbc892d6549a465
#   Stored in directory: /tmp/pip-ephem-wheel-cache-evzyvk3d/wheels/0d/7f/1d/ef1fb6071f11b18ea0090dbf9900d0c6788c9bf43a2a5818b5
# Successfully built bleurt-pytorch
# Installing collected packages: huggingface-hub, bleurt-pytorch
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# fastcoref 2.1.6 requires datasets>=2.5.2, which is not installed.

# Using cached wheel-0.45.1-py3-none-any.whl (72 kB)
# Building wheels for collected packages: alignscore
#   Building wheel for alignscore (pyproject.toml) ... done
#   Created wheel for alignscore: filename=alignscore-0.1.3-py3-none-any.whl size=18357 sha256=f20f3a1322d920d5e220d446719287d2dda5102950a712073cee23d785f045e4
#   Stored in directory: /tmp/pip-ephem-wheel-cache-xi73usjq/wheels/56/4e/2b/046d1e57776255e81e35c3903a3e67ddc8807b1b2dcb32b7de
# Successfully built alignscore
# Installing collected packages: wheel, werkzeug, tensorboard-data-server, six, protobuf, nvidia-cuda-nvrtc-cu11, markdown, lightning-utilities, jsonlines, grpcio, fsspec, tensorboard, nvidia-cuda-runtime-cu11, nvidia-cublas-cu11, nvidia-cudnn-cu11, torch, datasets, torchmetrics, pytorch-lightning, alignscore
#   Attempting uninstall: protobuf
#     Found existing installation: protobuf 5.29.3
#     Uninstalling protobuf-5.29.3:
#       Successfully uninstalled protobuf-5.29.3
#   Attempting uninstall: fsspec
#     Found existing installation: fsspec 2024.9.0
#     Uninstalling fsspec-2024.9.0:
#       Successfully uninstalled fsspec-2024.9.0
#   Attempting uninstall: torch
#     Found existing installation: torch 2.3.1
#     Uninstalling torch-2.3.1:
#       Successfully uninstalled torch-2.3.1
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torchvision 0.18.1 requires torch==2.3.1, but you have torch 1.13.1 which is incompatible.
# lmdeploy 0.6.1 requires torch<=2.3.1,>=2.0.0, but you have torch 1.13.1 which is incompatible.
# accelerate 1.4.0 requires torch>=2.0.0, but you have torch 1.13.1 which is incompatible.
# Successfully installed alignscore-0.1.3 datasets-2.21.0 fsspec-2024.6.1 grpcio-1.71.0 jsonlines-2.0.0 lightning-utilities-0.14.3 markdown-3.8 nvidia-cublas-cu11-11.10.3.66 nvidia-cuda-nvrtc-cu11-11.7.99 nvidia-cuda-runtime-cu11-11.7.99 nvidia-cudnn-cu11-8.5.0.96 protobuf-3.20.0 pytorch-lightning-1.9.5 six-1.17.0 tensorboard-2.19.0 tensorboard-data-server-0.7.2 torch-1.13.1 torchmetrics-1.5.2 werkzeug-3.1.3 wheel-0.45.1

#===============================================================================
# REFERENCE INFORMATION
#===============================================================================

log_section "Helpful SLURM Commands"

echo "  - cluster-info                 : Show cluster information"
echo "  - sinfo -N -l                  : List nodes and their status"
echo "  - squeue                       : Show job queue"
echo "  - scancel <JOB_ID>             : Cancel a specific job"
echo "  - scancel -u \$USER             : Cancel all your jobs"
echo "  - scontrol show job <JOB_ID>   : Show detailed job information"
echo "  - scontrol show node <NODE>    : Show detailed information about a node"
echo "  - squeue -u \$USER             : Show jobs for your user only"
echo "  - srun --ntasks=1 --pty bash   : Launch an interactive session on a compute node"
echo "  - sbatch <script.sh>           : Submit a batch job from a script"

#===============================================================================
# CLEANUP
#===============================================================================

log_section "Cleanup and Completion"

log_info "Deactivating virtual environment..."
deactivate

log_info "BLS environment setup completed successfully!"
log_info "Completed at $(date)"