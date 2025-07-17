import torch
from typing import Tuple
from unsloth import FastLanguageModel
from utils.helpers import setup_logging

logger = setup_logging()

def initialize_complexity_tokens(model, tokenizer, complexity_tokens):
    """Initialize complexity tokens with semantic meaning in embedding space."""
    # Add tokens to vocabulary first
    logger.info(f"Adding {len(complexity_tokens)} complexity tokens to vocabulary")
    num_added = tokenizer.add_tokens(complexity_tokens)

    if num_added < len(complexity_tokens):
        logger.warning(f"Only added {num_added}/{len(complexity_tokens)} tokens. Some may already exist in vocabulary.")

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model vocabulary size after adding tokens: {len(tokenizer)}")

    # Get embedding layer for manipulation
    embedding_layer = model.get_input_embeddings()
    
    # Define semantic anchors - words that represent the extremes of our complexity scale
    # Using multiple words gives us a more robust direction vector
    simple_words = ["simple", "basic", "elementary"]
    complex_words = ["complex", "advanced", "technical"]
    
    # Get embeddings for simple concepts (using only single-token words)
    simple_embeds = [embedding_layer.weight.data[tokenizer.encode(word, add_special_tokens=False)[0]] 
                    for word in simple_words 
                    if len(tokenizer.encode(word, add_special_tokens=False)) == 1]
    
    # Get embeddings for complex concepts (using only single-token words)
    complex_embeds = [embedding_layer.weight.data[tokenizer.encode(word, add_special_tokens=False)[0]] 
                     for word in complex_words 
                     if len(tokenizer.encode(word, add_special_tokens=False)) == 1]
    
    # Average embeddings to get anchor points
    simple_center = torch.stack(simple_embeds).mean(dim=0)
    complex_center = torch.stack(complex_embeds).mean(dim=0)
    
    # Create semantic direction vector from simple to complex
    # This vector represents the "direction of increasing complexity" in embedding space
    direction = complex_center - simple_center
    direction = direction / direction.norm()  # Normalize to unit vector
    
    # Get embedding statistics from existing model embeddings
    # This helps our new embeddings maintain similar statistical properties
    original_embeddings = embedding_layer.weight[:-num_added].detach()
    mean_embed = original_embeddings.mean(dim=0)
    std_norm = original_embeddings.norm(dim=1).std()
    
    # Scale direction vector to match model's embedding norms
    # This prevents our new tokens from having unusual magnitudes
    direction = direction * std_norm
    
    # Extract complexity levels from the config.complexity.range_min/max/step values
    # We use the token index rather than parsing the token text to avoid format assumptions
    total_tokens = len(complexity_tokens)
    # Normalize complexity levels to range [-1, 1]
    norm_levels = [(i / (total_tokens - 1) - 0.5) * 2 for i in range(total_tokens)]
    
    # Initialize each token embedding along the bidirectional complexity dimension
    for token, norm_level in zip(complexity_tokens, norm_levels):
        # Create embedding by starting at mean and moving along direction vector
        # The distance moved is proportional to the token's complexity level
        new_embedding = mean_embed + norm_level * direction
        
        # Assign the new embedding to the token
        token_id = tokenizer.convert_tokens_to_ids(token)
        embedding_layer.weight.data[token_id] = new_embedding
    
    return model, tokenizer
    
def setup_model_and_tokenizer(config):
    """Initialize model and tokenizer with Unsloth optimization."""
    logger.info(f"Loading model: {config.model.name}")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name,
        max_seq_length=config.data.max_length,
        dtype=None,  # Will be determined by precision flags
        load_in_4bit=config.model.load_in_4bit,
        device_map=config.model.device_map,
    )

    # Skip token initialization if not using control tokens or not adding tokens to vocab
    if not config.complexity.use_control_tokens or not config.complexity.add_tokens_to_vocab:
        return model, tokenizer
        
    # Add complexity tokens to vocabulary
    complexity_tokens = [
        config.complexity.token_format.format(level=bin_value) 
        for bin_value in range(
            config.complexity.range_min,
            config.complexity.range_max + config.complexity.range_step,
            config.complexity.range_step
        )
    ]
    
    # Test encoding BEFORE adding tokens
    logger.info("Testing tokenization BEFORE adding complexity tokens:")
    for token in complexity_tokens[:3]:  # Test first 3 tokens
        test_encoding = tokenizer.encode(token)
        logger.info(f"  {token} → {len(test_encoding)} tokens: {test_encoding}")
    
    # Initialize tokens using semantic embedding method
    model, tokenizer = initialize_complexity_tokens(
        model, tokenizer, complexity_tokens
    )
    
    # Test encoding AFTER adding tokens
    logger.info("Testing tokenization AFTER adding complexity tokens:")
    for token in complexity_tokens[:3]:  # Test first 3 tokens
        test_encoding = tokenizer.encode(token)
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"  {token} → {len(test_encoding)} tokens: {test_encoding}")
        logger.info(f"  Token ID for {token}: {token_id}")

        # Check if the token appears as a single token in the encoding
        token_in_encoding = token_id in test_encoding
        if token_in_encoding:
            logger.info(f"  ✓ {token} is treated as a single token (ID: {token_id})")
            continue
            
        logger.warning(f"  ✗ {token} is NOT treated as a single token!")
    
    return model, tokenizer

def apply_lora(model, tokenizer, config):
    """Apply LoRA to model."""
    logger.info(f"Applying LoRA: r={config.lora.r}, alpha={config.lora.alpha}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        use_gradient_checkpointing=config.model.use_gradient_checkpointing,
        random_state=config.seed,
        use_rslora=config.lora.use_rslora,
        loftq_config=None,
    )
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percent = trainable_params / total_params * 100 if total_params > 0 else 0
    
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}% of total)")
    
    return model