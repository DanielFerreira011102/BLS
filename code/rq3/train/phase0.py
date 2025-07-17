import argparse
from dataclasses import asdict
from pathlib import Path

# Configuration imports
from config import TrainingConfig, ModelConfig, DataConfig, ComplexityConfig, LoraConfig, TrainerConfig, WandbConfig

# Function imports
from data import load_complexity_data, prepare_training_data, create_dataset, CustomDataCollator
from model import setup_model_and_tokenizer, apply_lora
from callbacks import LoggingCallback, QualitativeTestCallback, run_qualitative_test
from trainer import ContextAwareSFTTrainer, create_training_args
from utils.helpers import setup_logging, set_seed, save_json

import wandb
from transformers import EarlyStoppingCallback

logger = setup_logging()

class ComplexityTrainer:
    """Trainer for complexity-controlled language models."""
    
    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.output_dir = self._setup_dirs()
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Set seed for reproducibility
        set_seed(config.seed)
    
    def _setup_dirs(self) -> Path:
        """Set up output directories."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "qualitative_tests").mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        save_json(asdict(self.config), output_dir / "config.json")
        
        return output_dir
    
    def setup_model(self):
        """Initialize and prepare the model."""
        self.model, self.tokenizer = setup_model_and_tokenizer(self.config)
        self.model = apply_lora(self.model, self.tokenizer, self.config)
        return self.model, self.tokenizer
    
    def load_data(self):
        """Load and prepare datasets."""
        df = load_complexity_data(
            self.config.data.path,
            self.config,
            self.config.data.test_mode, 
            self.config.data.test_sample_size, 
            self.config.seed
        )
        
        train_df, val_df = prepare_training_data(df, self.config)
        
        # Create datasets with the config
        self.train_dataset = create_dataset(train_df, self.tokenizer, self.config)
        self.val_dataset = create_dataset(val_df, self.tokenizer, self.config)
        
        return self.train_dataset, self.val_dataset
    
    def _setup_wandb(self):
        """Set up Weights & Biases if enabled."""
        if not self.config.wandb.use_wandb:
            return
            
        logger.info("Initializing Weights & Biases...")
        wandb.init(
            entity=self.config.wandb.entity, 
            project=self.config.wandb.project,
            name=self.config.wandb.run_name,
            config=asdict(self.config)
        )
        logger.info("Weights & Biases initialized")
    
    def _create_trainer(self):
        """Create and configure trainer."""
        training_args = create_training_args(self.config)
        
        # Set up callbacks
        callbacks = [
            LoggingCallback(),
            QualitativeTestCallback(
                model=self.model,
                tokenizer=self.tokenizer,
                config=self.config,
                output_dir=self.output_dir / "qualitative_tests"
            )
        ]
        
        # Add early stopping if configured
        if self.config.trainer.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.trainer.early_stopping_patience,
                    early_stopping_threshold=self.config.trainer.early_stopping_threshold
                )
            )
        
        # Set up data collator
        data_collator = CustomDataCollator(
            self.tokenizer,
            max_length=self.config.data.max_length
        )

        return ContextAwareSFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            data_collator=data_collator,
            callbacks=callbacks,
            context_aware_batching=self.config.trainer.context_aware_batching,
        )
    
    def train(self):
        """Run the complete training pipeline."""
        # Setup model first (needed for tokenizer)
        self.setup_model()
        
        # Load data (needs tokenizer)
        self.load_data()
        
        # Setup W&B before trainer creation
        self._setup_wandb()
        
        # Create trainer
        trainer = self._create_trainer()
        
        # Run baseline evaluation
        logger.info("Running baseline qualitative test...")
        baseline_results = run_qualitative_test(
            self.model, 
            self.tokenizer, 
            self.config, 
            step=0
        )
        save_json(
            baseline_results, 
            self.output_dir / "qualitative_tests" / "test_step_0.json"
        )
        
        # Check for existing checkpoints
        resume_checkpoint = None
        if self.config.trainer.resume_from_checkpoint:
            checkpoints = sorted(Path(self.config.output_dir).glob("checkpoint-*"))
            if checkpoints:
                latest_checkpoint = checkpoints[-1]
                logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                resume_checkpoint = str(latest_checkpoint)
        
        # Train model
        logger.info("Starting training...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        
        # Save final model and run evaluation
        self._save_results(trainer)
        
        return self.model, self.tokenizer
    
    def _save_results(self, trainer):
        """Save final model and run evaluation."""
        logger.info("Saving final model...")
        trainer.save_model(self.output_dir / "final_model")
        self.tokenizer.save_pretrained(self.output_dir / "final_model")
        
        # Run final evaluation
        logger.info("Running final qualitative test...")
        final_results = run_qualitative_test(
            self.model, 
            self.tokenizer, 
            self.config
        )
        save_json(
            final_results, 
            self.output_dir / "qualitative_tests" / "test_step_final.json"
        )
        
        logger.info(f"Training completed, model saved to {self.output_dir / 'final_model'}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train complexity-controlled language model")
    
    # Create default config
    default_config = TrainingConfig()
    
    # Add argument groups for better organization
    path_args = parser.add_argument_group("Paths")
    model_args = parser.add_argument_group("Model Configuration")
    data_args = parser.add_argument_group("Data Configuration")
    complexity_args = parser.add_argument_group("Complexity Control")
    lora_args = parser.add_argument_group("LoRA Configuration")
    training_args = parser.add_argument_group("Training")
    wandb_args = parser.add_argument_group("Weights & Biases")
    
    # General arguments
    parser.add_argument("--seed", type=int, default=default_config.seed,
                      help="Random seed")
    path_args.add_argument("--output-dir", type=str, default=default_config.output_dir,
                      help="Directory to save model and outputs")
    
    # Model arguments
    model_args.add_argument("--model-name", type=str, default=default_config.model.name,
                      help="Base model to fine-tune")
    model_args.add_argument("--load-in-4bit", action="store_true", default=default_config.model.load_in_4bit,
                      help="Load model in 4-bit precision")
    model_args.add_argument("--device-map", type=str, default=default_config.model.device_map,
                      help="Device map for model distribution")
    model_args.add_argument("--use-gradient-checkpointing", action="store_true",
                      default=default_config.model.use_gradient_checkpointing,
                      help="Enable gradient checkpointing")
    
    # Data arguments
    data_args.add_argument("--data-path", type=str, required=True,
                      help="Path to complexity data CSV")
    data_args.add_argument("--max-length", type=int, default=default_config.data.max_length,
                      help="Maximum sequence length for input")
    data_args.add_argument("--validation-split", type=float, default=default_config.data.validation_split,
                      help="Fraction of data to use for validation")
    data_args.add_argument("--test-mode", action="store_true", default=default_config.data.test_mode,
                      help="Enable test mode with reduced dataset")
    data_args.add_argument("--test-sample-size", type=int, default=default_config.data.test_sample_size,
                      help="Number of samples to use in test mode")
    
    # Complexity arguments
    complexity_args.add_argument("--add-complexity-tokens", action="store_true",
                      dest="add_complexity_tokens",
                      default=default_config.complexity.add_tokens_to_vocab,
                      help="Add complexity tokens to model vocabulary")
    complexity_args.add_argument("--use-control-tokens", action="store_true",
                      dest="use_control_tokens",
                      default=default_config.complexity.use_control_tokens,
                      help="Use control tokens instead of natural language prompts")
    complexity_args.add_argument("--complexity-column", type=str,
                      choices=["complexity_score_quantile", "complexity_score_percentile_discretized"],
                      default=default_config.complexity.complexity_column,
                      help="Column to use for complexity scores")
    complexity_args.add_argument("--complexity-token-format", type=str,
                      default=default_config.complexity.token_format,
                      help="Format string for complexity tokens")
    complexity_args.add_argument("--prompt-format", type=str,
                      default=default_config.complexity.prompt_format,
                      help="Format string for natural language complexity prompts")
    complexity_args.add_argument("--complexity-range-min", type=int,
                      default=default_config.complexity.range_min,
                      help="Minimum complexity level")
    complexity_args.add_argument("--complexity-range-max", type=int,
                      default=default_config.complexity.range_max,
                      help="Maximum complexity level")
    complexity_args.add_argument("--complexity-range-step", type=int,
                      default=default_config.complexity.range_step,
                      help="Step size for complexity levels")
    
    # LoRA arguments
    lora_args.add_argument("--lora-r", type=int, default=default_config.lora.r,
                      help="LoRA rank")
    lora_args.add_argument("--lora-alpha", type=int, default=default_config.lora.alpha,
                      help="LoRA alpha parameter")
    lora_args.add_argument("--lora-dropout", type=float, default=default_config.lora.dropout,
                      help="LoRA dropout rate")
    
    # Training arguments
    training_args.add_argument("--batch-size", type=int, default=default_config.trainer.batch_size,
                      help="Training batch size")
    training_args.add_argument("--context-aware-batching", action="store_true",
                      default=default_config.trainer.context_aware_batching,
                      help="Enable context-aware batching")
    training_args.add_argument("--learning-rate", type=float, default=default_config.trainer.learning_rate,
                      help="Learning rate")
    training_args.add_argument("--num-epochs", type=int, default=default_config.trainer.num_epochs,
                      help="Number of training epochs")
    training_args.add_argument("--max-steps", type=int, default=default_config.trainer.max_steps,
                      help="Maximum number of training steps")
    training_args.add_argument("--warmup-ratio", type=float, default=default_config.trainer.warmup_ratio,
                      help="Ratio of steps for learning rate warmup")
    training_args.add_argument("--gradient-accumulation-steps", type=int,
                      default=default_config.trainer.gradient_accumulation_steps,
                      help="Number of gradient accumulation steps")
    training_args.add_argument("--weight-decay", type=float, default=default_config.trainer.weight_decay,
                      help="Weight decay for optimizer")
    training_args.add_argument("--save-steps", type=int, default=default_config.trainer.save_steps,
                      help="Steps between model checkpoint saves")
    training_args.add_argument("--eval-steps", type=int, default=default_config.trainer.eval_steps,
                      help="Steps between evaluations")
    training_args.add_argument("--logging-steps", type=int, default=default_config.trainer.logging_steps,
                      help="Steps between logging")
    training_args.add_argument("--qualitative-test-steps", type=int,
                      default=default_config.trainer.qualitative_test_steps,
                      help="Steps between qualitative tests")
    training_args.add_argument("--early-stopping-patience", type=int,
                      default=default_config.trainer.early_stopping_patience,
                      help="Number of evaluations with no improvement before stopping")
    training_args.add_argument("--resume-from-checkpoint", action="store_true",
                      default=default_config.trainer.resume_from_checkpoint,
                      help="Resume training from latest checkpoint if available")
    
    # W&B arguments
    wandb_args.add_argument("--use-wandb", action="store_true", default=default_config.wandb.use_wandb,
                      help="Enable Weights & Biases logging")
    wandb_args.add_argument("--wandb-entity", type=str, default=default_config.wandb.entity,
                      help="Weights & Biases entity name")
    wandb_args.add_argument("--wandb-project", type=str, default=default_config.wandb.project,
                      help="Weights & Biases project name")
    wandb_args.add_argument("--wandb-run-name", type=str, default=default_config.wandb.run_name,
                      help="Weights & Biases run name")
    
    args = parser.parse_args()
    
    # Build config from args
    return build_config_from_args(args)

def build_config_from_args(args):
    """Build the configuration from parsed arguments."""
    # Initialize configs with default values
    model_config = ModelConfig(
        name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )
    
    data_config = DataConfig(
        path=args.data_path,
        max_length=args.max_length,
        validation_split=args.validation_split,
        test_mode=args.test_mode,
        test_sample_size=args.test_sample_size
    )
    
    complexity_config = ComplexityConfig(
        add_tokens_to_vocab=args.add_complexity_tokens,
        use_control_tokens=args.use_control_tokens,
        token_format=args.complexity_token_format,
        prompt_format=args.prompt_format,
        complexity_column=args.complexity_column,
        range_min=args.complexity_range_min,
        range_max=args.complexity_range_max,
        range_step=args.complexity_range_step
    )
    
    lora_config = LoraConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    
    trainer_config = TrainerConfig(
        batch_size=args.batch_size,
        context_aware_batching=args.context_aware_batching,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        qualitative_test_steps=args.qualitative_test_steps,
        early_stopping_patience=args.early_stopping_patience,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    wandb_config = WandbConfig(
        use_wandb=args.use_wandb,
        entity=args.wandb_entity,
        project=args.wandb_project,
        run_name=args.wandb_run_name
    )
    
    # Create master config
    config = TrainingConfig(
        model=model_config,
        data=data_config,
        complexity=complexity_config,
        lora=lora_config,
        trainer=trainer_config,
        wandb=wandb_config,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    return config

def main():
    """Main entry point."""
    config = parse_args()
    trainer = ComplexityTrainer(config)
    trainer.train()
    return 0

if __name__ == "__main__":
    main()