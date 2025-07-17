import torch
from pathlib import Path
from typing import Dict, Any, List
from transformers import TrainerCallback
from utils.helpers import setup_logging, save_json

logger = setup_logging()

class LoggingCallback(TrainerCallback):
    """Callback for enhanced logging during training with basic metrics."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics in a readable format."""
        if not logs:
            return
            
        # Format step information
        step_info = f"Step {state.global_step}"
        if state.max_steps:
            step_info += f"/{state.max_steps}"
            
        # Prepare metrics text parts
        metrics_parts = []
        
        # Log training metrics
        if "loss" in logs:
            metrics_parts.append(f"loss: {logs['loss']:.4f}")
            
        # Log evaluation metrics
        if "eval_loss" in logs:
            metrics_parts.append(f"eval_loss: {logs['eval_loss']:.4f}")
            
        # Log learning rate
        if "learning_rate" in logs:
            metrics_parts.append(f"lr: {logs['learning_rate']:.6f}")
        
        # Log all metrics
        if metrics_parts:
            logger.info(f"{step_info} - {' | '.join(metrics_parts)}")

class QualitativeTestCallback(TrainerCallback):
    """Callback to run qualitative tests during training."""
    
    def __init__(self, model, tokenizer, config, output_dir):
        """Initialize callback with model, tokenizer and configuration."""
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Run tests at specified steps."""
        # Skip if step is 0 or not a testing step
        if state.global_step == 0 or state.global_step % self.config.trainer.qualitative_test_steps != 0:
            return
        
        logger.info(f"Qualitative test at step {state.global_step}")
        results = run_qualitative_test(
            self.model, 
            self.tokenizer, 
            self.config, 
            state.global_step
        )
        
        save_json(results, self.output_dir / f"test_step_{state.global_step}.json")

def run_qualitative_test(model, tokenizer, config, step=None) -> List[Dict[str, Any]]:
    """Generate sample outputs at different complexity levels."""
    results = []
    
    # Ensure model is in eval mode
    model.eval()
    
    # Process each prompt and complexity level
    for prompt in config.complexity.test_prompts:
        for level in config.complexity.test_levels:
            # Format prompt based on configuration
            if config.complexity.use_control_tokens:
                # Use complexity tokens
                complexity_token = config.complexity.token_format.format(level=level)
                user_message = f"{complexity_token} {prompt}"
            else:
                # Use natural language prompts
                user_message = config.complexity.prompt_format.format(
                    level=level,
                    question=prompt
                )
                
            # Format prompt using chat template
            messages = [{"role": "user", "content": user_message}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Generate text
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
            
            # Decode only the newly generated tokens (not the prompt)
            new_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Store result
            results.append({
                "step": step,
                "prompt": prompt,
                "complexity_level": level,
                "generated_text": generated_text.strip()
            })
    
    # Return to training mode
    model.train()
    
    return results