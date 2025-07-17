import torch
import gc
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from unsloth import FastLanguageModel
from utils.helpers import setup_logging, save_json, load_json
from transformers import AutoTokenizer
from peft import PeftModel
import anthropic

# Setup logging
logger = setup_logging()

def load_questions(data_path, num_questions=None, skip_first=0):
    """Load questions from the dataset, with options to skip and limit."""
    # Use the helper function to load JSON
    data = load_json(data_path)
    
    # Get the questions from the data
    questions = []
    for i, item in enumerate(data['qa_pairs']):
        # Skip the first 'skip_first' questions
        if i < skip_first:
            continue
            
        # Break once we have enough questions (if limit is set)
        if num_questions is not None and len(questions) >= num_questions:
            break
            
        questions.append(item['question'])
    
    logger.info(f"Loaded all {len(questions)} questions from dataset")    
    return questions

def generate_baseline_responses(questions, complexity_levels, model_name):
    """Generate responses using the baseline model."""
    results = []
    
    logger.info(f"Loading baseline model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="auto"
    )
    
    for question in tqdm(questions, desc="Processing questions (baseline)"):
        for level in tqdm(complexity_levels, desc=f"Complexity levels for: {question[:30]}...", leave=False):
            # Format using chat template to match training method
            messages = [{
                "role": "user", 
                "content": f"Answer the following medical question with a complexity score of {level} out of 100: {question}"
            }]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate text
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
            
            # Decode only the generated tokens
            new_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Store results
            results.append({
                "question": question,
                "complexity_level": level,
                "model": "baseline",
                "response": generated_text.strip()
            })
    
    # Clear memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def generate_finetuned_responses(questions, complexity_levels, model_path, base_model_name, use_control_codes=True):
    """Generate responses using the fine-tuned model."""
    results = []
    
    logger.info(f"Loading fine-tuned model from: {model_path}")
    logger.info(f"Using {'control codes' if use_control_codes else 'natural language instructions'}")
    
    # Load the saved tokenizer first
    tokenizer_path = Path(model_path)
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    logger.info(f"Loaded tokenizer with {len(tokenizer)} tokens")
    
    # First, load the base model
    logger.info(f"Loading base model: {base_model_name}")
    model, _ = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Resize token embeddings to match the loaded tokenizer
    model.resize_token_embeddings(len(tokenizer))
    
    # Now load the LoRA weights from the fine-tuned model
    logger.info(f"Loading LoRA weights from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    
    for question in tqdm(questions, desc="Processing questions (fine-tuned)"):
        for level in tqdm(complexity_levels, desc=f"Complexity levels for: {question[:30]}...", leave=False):
            # Format the prompt based on whether to use control codes
            if use_control_codes:
                # Apply control token format as used in training
                complexity_token = f"<COMPLEXITY_{level}>"
                user_message = f"{complexity_token} {question}"
            else:
                # Use natural language instruction
                user_message = f"Answer the following medical question with a complexity score of {level} out of 100: {question}"
            
            # Format using chat template to match training
            messages = [{"role": "user", "content": user_message}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate text
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
            
            # Decode only the generated tokens
            new_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Store results
            results.append({
                "question": question,
                "complexity_level": level,
                "model": "finetuned" if use_control_codes else "finetuned-nl",
                "response": generated_text.strip()
            })
    
    # Clear memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def generate_claude_responses(questions, complexity_levels, api_key=None):
    """Generate responses using the Claude 3.7 Sonnet API."""
    results = []
    
    logger.info("Starting Claude 3.7 API evaluation")
    
    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Process each question individually
    for i in tqdm(range(len(questions)), desc="Processing questions (Claude API)"):
        question = questions[i]

        # Create requests for all complexity levels of this question
        requests = []
        for level in complexity_levels:
            requests.append({
                "custom_id": f"level_{level}",
                "params": {
                    "model": "claude-3-7-sonnet-20250219",
                    "max_tokens": 500,
                    "temperature": 0.7,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Answer the following medical question with a complexity score of {level} out of 100: {question}"
                        }
                    ]
                }
            })
        
        # Submit the batch for this question (all complexity levels)
        batch = client.beta.messages.batches.create(requests=requests)
        batch_id = batch.id

        # Wait for the batch to complete processing
        while True:
            batch_status = client.beta.messages.batches.retrieve(batch_id)
            if batch_status.processing_status == "ended":
                break
            time.sleep(2)  # Wait 2 seconds before checking again

        # Process results
        for result in client.beta.messages.batches.results(batch.id):
            custom_id = result.custom_id
            level_value = int(custom_id.split("_")[1])
            if result.result.type == "succeeded":
                results.append({
                    "question": question,
                    "complexity_level": level_value,
                    "model": "claude-api",
                    "response": result.result.message.content[0].text
                })
    
    return results

def generate_fewshot_responses(questions, complexity_levels, model_name, examples_data, num_examples=5):
    """Generate responses using few-shot examples with the baseline model."""
    results = []
    
    logger.info(f"Loading baseline model for few-shot generation: {model_name}")
    logger.info(f"Using {num_examples} examples for few-shot learning")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map="auto"
    )
    
    # Parse the examples data into appropriate format
    examples = examples_data["examples"]
    example_question = examples_data["question"]
    
    for question in tqdm(questions, desc="Processing questions (few-shot)"):
        for level in tqdm(complexity_levels, desc=f"Complexity levels for: {question[:30]}...", leave=False):
            # Select the closest examples based on complexity level
            selected_examples = select_examples_for_level(examples, level, num_examples)
            
            # Create the few-shot prompt
            few_shot_prompt = create_few_shot_prompt(example_question, selected_examples, question, level)
            
            # Format using chat template
            messages = [{"role": "user", "content": few_shot_prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Generate text
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                )
            
            # Decode only the generated tokens
            new_tokens = outputs[0][input_length:]
            generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Store results
            results.append({
                "question": question,
                "complexity_level": level,
                "model": "few-shot",
                "response": generated_text.strip()
            })
    
    # Clear memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def select_examples_for_level(examples, target_level, num_examples=5):
    """Select appropriate examples that are closest to the target complexity level."""
    # Sort examples by how close they are to the target level
    sorted_examples = sorted(examples, key=lambda x: abs(x['complexity'] - target_level))
    
    # Take the closest examples
    return sorted_examples[:num_examples]

def create_few_shot_prompt(example_question, examples, question, target_level):
    """Create a few-shot prompt with examples at different complexity levels."""
    prompt = f"I'll show you a medical question answered at different complexity levels, then ask you to answer a new question at a specific complexity level.\n\n"
    prompt += f"EXAMPLE QUESTION: {example_question}\n\n"
    
    # Add examples
    for i, example in enumerate(examples):
        prompt += f"COMPLEXITY LEVEL {example['complexity']}:\n{example['response']}\n\n"
    
    # Add the target question and level
    prompt += f"Now, please answer the following question at COMPLEXITY LEVEL {target_level}:\n{question}"
    
    return prompt

def load_examples_data(examples_path):
    """Load few-shot examples data from file or create from default examples."""
    if examples_path and Path(examples_path).exists():
        logger.info(f"Loading few-shot examples from: {examples_path}")
        return load_json(examples_path)
    else:
        logger.info("Using default few-shot examples")
        # Default examples
        example_question = "Is Diabetes mellitus type 1 inherited?"
        examples = [
            {
                "complexity": 0,
                "response": "Diabetes type 1 isn't something you get just because someone in your family had it. But some people might be more likely to get it because of their family. Even if you have those genes, you might not get sick. It's not just about genes; other things matter too. If someone in your family has it, you might be more at risk, but you can talk to a doctor to learn more."
            },
            {
                "complexity": 10,
                "response": "Diabetes type 1 isn't directly inherited, but having a family history can increase your risk. Some people are more likely to develop it because of certain genes, like those related to the immune system. However, not everyone with these genes will get diabetes. Other factors also play a role. If a parent or sibling has type 1 diabetes, your risk is higher, but it's still not certain. Genetic testing can help assess your risk, and talking to a healthcare provider can provide more clarity."
            },
            {
                "complexity": 40,
                "response": "Diabetes mellitus type 1 is not directly inherited, but there is a genetic predisposition. Certain genetic factors, such as specific HLA haplotypes, can increase the risk. However, these genetic markers are common in the general population, and only about 5% of individuals with them develop the condition. Family history can elevate risk, especially if a parent or sibling has the disease. Genetic testing may help determine risk, but the condition's multifactorial nature makes prediction difficult. Consulting a healthcare provider is recommended for personalized advice."
            },
            {
                "complexity": 60,
                "response": "Diabetes mellitus type 1 is influenced by genetic predisposition rather than direct inheritance. Specific HLA gene variants and their haplotypes contribute to an increased risk, though they are prevalent in the general population. Approximately 5% of individuals with these genetic factors develop the condition. Family history increases risk, particularly if a first-degree relative is affected. The absence of a clear inheritance pattern complicates risk prediction. Genetic testing can provide insights, but consultation with a healthcare professional is essential for personalized assessment and management."
            },
            {
                "complexity": 90,
                "response": "Diabetes mellitus type 1 exhibits a complex genetic architecture without a straightforward mendelian inheritance pattern. While specific HLA class II haplotypes significantly contribute to susceptibility, their presence in the general population and the low penetrance (approximately 5%) indicate a multifactorial etiology. Familial clustering is observed, with elevated risk for first-degree relatives, though it remains non-deterministic. The condition's pathogenesis involves immune dysregulation, pancreatic islet autoimmunity, and environmental triggers. Genetic screening may identify predisposed individuals, but clinical application requires integration with epidemiological and immunogenetic data. Specialist consultation is recommended for precise risk assessment and management strategies."
            }
        ]
        return {"question": example_question, "examples": examples}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate language models with complexity control.")
    
    # Add arguments
    parser.add_argument("--data-path", type=str, 
                        default="/data/home/djbf/storage/bls/resources/datasets/claude/health_qa_20250222_205257.json",
                        help="Path to the dataset")
    parser.add_argument("--base-model", type=str, 
                        default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                        help="Base model name")
    parser.add_argument("--ft-model", type=str, 
                        default="/data/home/djbf/storage/bls/rq3/outputs/models/complexity_20250502_055712/final_model",
                        help="Fine-tuned model path")
    parser.add_argument("--output-dir", type=str, 
                        default="./complexity_evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--num-questions", type=int, 
                        default=None,
                        help="Number of questions to evaluate (None = all questions)")
    parser.add_argument("--skip-first", type=int, 
                        default=0,
                        help="Number of questions to skip from the beginning")
    parser.add_argument("--eval-baseline", action="store_true",
                        help="Run baseline model evaluation")
    parser.add_argument("--eval-finetuned", action="store_true",
                        help="Run fine-tuned model evaluation")
    parser.add_argument("--eval-fewshot", action="store_true",
                        help="Run few-shot model evaluation")
    parser.add_argument("--eval-claude", action="store_true",
                        help="Run Claude 3.7 API evaluation")
    parser.add_argument("--no-control-codes", action="store_true",
                        help="Use natural language instructions instead of control codes for fine-tuned models")
    parser.add_argument("--claude-api-key", type=str, 
                        default=None,
                        help="Claude API key (defaults to ANTHROPIC_API_KEY environment variable)")
    parser.add_argument("--examples-path", type=str, 
                        default=None,
                        help="Path to few-shot examples data (JSON)")
    parser.add_argument("--num-examples", type=int, 
                        default=3,
                        help="Number of examples to use for few-shot (default: 3)")
    parser.add_argument("--complexity-min", type=int, default=0,
                        help="Minimum complexity level")
    parser.add_argument("--complexity-max", type=int, default=100,
                        help="Maximum complexity level")
    parser.add_argument("--complexity-step", type=int, default=5,
                        help="Step size for complexity levels")

    args = parser.parse_args()
    
    # If none of the eval flags are specified, evaluate all methods except Claude
    # (since Claude costs money, it should be explicitly enabled)
    if not args.eval_baseline and not args.eval_finetuned and not args.eval_fewshot and not args.eval_claude:
        args.eval_baseline = True
        args.eval_finetuned = True
        args.eval_fewshot = True
        
    return args

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Settings
    complexity_levels = list(range(args.complexity_min, args.complexity_max + 1, args.complexity_step))
    
    # Load data
    questions = load_questions(args.data_path, args.num_questions, args.skip_first)
    
    all_results = []
    
    # Generate responses with baseline model if requested
    if args.eval_baseline:
        logger.info("Starting baseline model evaluation")
        baseline_results = generate_baseline_responses(questions, complexity_levels, args.base_model)
        # Save intermediate results
        save_json(baseline_results, output_dir / "baseline_results.json")
        logger.info("Completed baseline model evaluation")
        all_results.extend(baseline_results)
    
    # Generate responses with fine-tuned model if requested
    if args.eval_finetuned:
        logger.info("Starting fine-tuned model evaluation")
        finetuned_results = generate_finetuned_responses(
            questions, 
            complexity_levels, 
            args.ft_model, 
            args.base_model,
            not args.no_control_codes  # Pass the opposite of no_control_codes
        )
        # Save intermediate results
        save_json(finetuned_results, output_dir / "finetuned_results.json")
        logger.info("Completed fine-tuned model evaluation")
        all_results.extend(finetuned_results)
    
    # Generate responses with few-shot approach if requested
    if args.eval_fewshot:
        logger.info("Starting few-shot model evaluation")
        # Load examples data
        examples_data = load_examples_data(args.examples_path)
        # Save the examples used
        save_json(examples_data, output_dir / "examples_used.json")
        
        fewshot_results = generate_fewshot_responses(
            questions, 
            complexity_levels, 
            args.base_model, 
            examples_data, 
            args.num_examples
        )
        # Save intermediate results
        save_json(fewshot_results, output_dir / "fewshot_results.json")
        logger.info("Completed few-shot model evaluation")
        all_results.extend(fewshot_results)
    
    # Generate responses with Claude API if requested
    if args.eval_claude:
        logger.info("Starting Claude 3.7 API evaluation")
        # Use provided API key or get from environment
        api_key = args.claude_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.error("No Claude API key provided. Set --claude-api-key or ANTHROPIC_API_KEY environment variable.")
            return
            
        claude_results = generate_claude_responses(questions, complexity_levels, api_key)
        # Save intermediate results
        save_json(claude_results, output_dir / "claude_results.json")
        logger.info("Completed Claude 3.7 API evaluation")
        all_results.extend(claude_results)
    
    # Save combined results if we have any
    if all_results:
        save_json(all_results, output_dir / "all_results.json")
        
        # Also save as CSV for easier analysis
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "all_results.csv", index=False)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()