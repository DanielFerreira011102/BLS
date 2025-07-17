from torch.utils.data import DataLoader
from trl import SFTTrainer, SFTConfig
from data import ContextAwareSampler
from utils.helpers import setup_logging

logger = setup_logging()

class ContextAwareSFTTrainer(SFTTrainer):
    """SFTTrainer with context-aware batching."""
    
    def __init__(self, *args, context_aware_batching=False, **kwargs):
        # Store context_aware_batching as a trainer attribute
        self.context_aware_batching = context_aware_batching
        super().__init__(*args, **kwargs)
    
    def get_train_dataloader(self) -> DataLoader:
        """Use context-aware batching if enabled."""
        if not self.context_aware_batching:
            return super().get_train_dataloader()
            
        # Create context-aware sampler
        logger.info("Using context-aware batching for training")
        sampler = ContextAwareSampler(self.train_dataset)
        
        # Create dataloader with sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )

def create_training_args(config) -> SFTConfig:
    """Create SFT training arguments from config."""
    # Set default logging directory
    logging_dir = f"{config.output_dir}/logs"
    
    # Determine report_to based on use_wandb config
    report_to = ["wandb"] if config.wandb.use_wandb else ["tensorboard"]
    
    return SFTConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.trainer.batch_size,
        per_device_eval_batch_size=config.trainer.batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        learning_rate=config.trainer.learning_rate,
        num_train_epochs=config.trainer.num_epochs,
        warmup_ratio=config.trainer.warmup_ratio,
        weight_decay=config.trainer.weight_decay,

        # Precision settings
        fp16=config.trainer.fp16,
        bf16=config.trainer.bf16,
        
        # Logging settings
        logging_steps=config.trainer.logging_steps,
        logging_dir=logging_dir,
        logging_strategy="steps",
        logging_first_step=True,
        log_level="info",
        
        # Evaluation settings
        eval_strategy="steps",
        save_strategy="steps",
        max_steps=config.trainer.max_steps,
        eval_steps=config.trainer.eval_steps,
        save_steps=config.trainer.save_steps,
        save_total_limit=config.trainer.save_total_limit,
        load_best_model_at_end=config.trainer.load_best_model_at_end,
        metric_for_best_model=config.trainer.metric_for_best_model,
        greater_is_better=config.trainer.greater_is_better,
        
        # Training settings
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        packing=False,
        remove_unused_columns=False,
        report_to=report_to,
        seed=config.seed,
        max_seq_length=config.data.max_length,
        include_inputs_for_metrics=True,
        completion_only_loss=config.trainer.completion_only_loss,
    )