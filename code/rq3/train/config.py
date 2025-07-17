from dataclasses import dataclass, field
from typing import List, Optional, Union
from datetime import datetime
from unsloth import is_bfloat16_supported
import dataclasses

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    load_in_4bit: bool = True
    device_map: str = "auto"
    use_gradient_checkpointing: Union[bool, str] = "unsloth"

@dataclass
class DataConfig:
    """Data processing configuration."""
    path: str = "complexity_data.csv"
    max_length: int = 1024
    validation_split: float = 0.05
    test_mode: bool = False
    test_sample_size: int = 1000

@dataclass
class ComplexityConfig:
    """Complexity control configuration."""
    add_tokens_to_vocab: bool = False  # Default to false since natural language is default
    use_control_tokens: bool = False  # Default to using natural language prompts
    token_format: str = "<COMPLEXITY_{level}>"
    prompt_format: str = "Answer the following question with a complexity score of {level} out of 100: {question}"
    complexity_column: str = "complexity_score_quantile"  # Can be "complexity_score_quantile" or "complexity_score_percentile_discretized"
    range_min: int = 0
    range_max: int = 100
    range_step: int = 5
    test_levels: List[int] = field(default_factory=lambda: [0, 25, 50, 75, 100])
    test_prompts: List[str] = field(default_factory=lambda: [
        "How is strep throat treated?",
        "Are the ADA hemoglobin A(1c) criteria relevant for the diagnosis of type 2 diabetes in youth?",
        "Explain how the immune system works."
    ])

@dataclass
class LoraConfig:
    """LoRA adaptation configuration."""
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    use_rslora: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainerConfig:
    """Training process configuration."""
    batch_size: int = 16
    context_aware_batching: bool = True
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    fp16: bool = not is_bfloat16_supported()
    bf16: bool = is_bfloat16_supported()
    max_steps: int = 32000
    save_steps: int = 1000
    eval_steps: int = 1000
    logging_steps: int = 100
    qualitative_test_steps: int = 500
    save_total_limit: int = 3
    completion_only_loss: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    load_best_model_at_end: bool = True
    resume_from_checkpoint: bool = False
    
@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    use_wandb: bool = True
    entity: str = "caa-overwatch"
    project: str = "complexity-llama"
    run_name: Optional[str] = f"run-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

@dataclass
class TrainingConfig:
    """Master configuration that contains sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    complexity: ComplexityConfig = field(default_factory=ComplexityConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    
    # Flat configuration for general settings
    output_dir: str = "./complexity_model"
    seed: int = 42
    
    def to_dict(self):
        """Convert config to dictionary, handling nested dataclasses."""
        return dataclasses.asdict(self)