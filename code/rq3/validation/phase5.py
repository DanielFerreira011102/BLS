import argparse
import pandas as pd
import numpy as np
import anthropic
import logging
import re
import json
import time
from pathlib import Path
from tqdm import tqdm
from utils.helpers import setup_logging, save_json

# Initialize logging
logger = setup_logging()


class ClaudeEvaluator:
    """Uses Claude API to evaluate medical responses across key dimensions."""

    def __init__(self, api_key=None):
        self.client = anthropic.Anthropic(api_key=api_key)

        # Evaluation dimensions and their descriptions
        self.dimensions = {
            "understandability": "How easy was this to understand?",
            "usefulness": "How useful was this answer to you?",
            "clarity": "Did you feel confused by any part of the text?",
            "relevance": "Does this answer address the question?",
            "factuality": "Is the answer factually correct?"
        }

        # Scale for scoring from 1 to 5
        self.scale = {
            1: "Poor",
            2: "Fair",
            3: "Good",
            4: "Very Good",
            5: "Excellent"
        }

        # Define simulated user personas by health literacy level
        self.personas = {
            "low": "You are a person with low health literacy evaluating medical information. You have no medical training and rely on everyday language to understand health topics. You struggle with medical jargon and need simple, clear explanations.",
            "medium": "You are a person with moderate health literacy evaluating medical information. You have some familiarity with common medical terms through personal experience, general education, or caring for family members. You can understand basic medical concepts but may struggle with highly technical information.",
            "high": "You are a healthcare professional or medical student evaluating medical information. You have extensive medical training and are comfortable with medical terminology, clinical concepts, and evidence-based practice."
        }

    def generate_evaluation_prompt(self, question, response, persona_prompt):
        """Constructs a structured evaluation prompt for Claude for a given persona."""
        return f"""
{persona_prompt}

You must evaluate the medical answer strictly from your own perspective and level of health literacy. Do not try to judge it from a general or professional viewpoint unless that matches your background.
Your task is to score the answer across five dimensions on a scale from 1 to 5, where 1 is the lowest and 5 is the highest.

The five dimensions and their levels (1 to 5) are defined as follows:

**Understandability**:
- 1: Very difficult to understand, confusing language or concepts
- 2: Somewhat difficult, requires effort to follow
- 3: Moderately understandable, generally clear
- 4: Easy to understand, well-explained concepts
- 5: Extremely clear and accessible for the intended audience

**Usefulness**:
- 1: Not helpful, lacks practical value
- 2: Minimally helpful, limited practical application
- 3: Moderately useful, provides some actionable information
- 4: Very useful, offers clear guidance or valuable insights
- 5: Extremely useful, highly actionable and comprehensive

**Clarity**:
- 1: Very confusing, many unclear or ambiguous parts
- 2: Somewhat confusing, several unclear elements
- 3: Generally clear with minor confusing aspects
- 4: Clear and well-structured, easy to follow
- 5: Exceptionally clear, no confusing elements

**Relevance**:
- 1: Does not address the question, completely off-topic
- 2: Minimally relevant, partially addresses the question
- 3: Moderately relevant, addresses main aspects of the question
- 4: Highly relevant, directly addresses the question well
- 5: Perfectly relevant, comprehensively addresses all aspects

**Factuality**:
- 1: Contains significant medical inaccuracies or misinformation
- 2: Contains some questionable or potentially inaccurate information
- 3: Generally accurate with minor issues or omissions
- 4: Medically accurate and reliable information
- 5: Completely accurate, evidence-based, and up-to-date

---
Question: {question}
Answer: {response}
---

Respond using only the following JSON format, without any additional text or explanations:
```json
{{
    "reasoning": "Brief reasoning for scores (optional, not required)",
    "understandability": 1-5,
    "usefulness": 1-5,
    "clarity": 1-5,
    "relevance": 1-5,
    "factuality": 1-5
}}
```
"""

    def parse_response(self, raw_text):
        """Attempts to extract a valid JSON object from Claude's output."""
        # Clean up the response text
        text = raw_text.strip()

        # Remove leading code blocks or backticks
        if text.startswith('```json'):
            text = text[7:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()
        if text.startswith('`'):
            text = text[1:].strip()

        # Remove any trailing newlines or spaces
        if text.endswith('```'):
            text = text[:-3].strip()
        if text.endswith('`'):
            text = text[:-1].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from response: {text}")
            # Fallback to regex extraction if JSON parsing fails
            return self.extract_scores_with_regex(text)

    def extract_scores_with_regex(self, response_text):
        """Extracts scores from Claude's response using regex."""
        scores = {dim: None for dim in self.dimensions.keys()}
        
        # Regex to find scores in the format "dimension: score"
        pattern = r'(understandability|usefulness|clarity|relevance|factuality)\s*:\s*(\d)'
        matches = re.findall(pattern, response_text, re.IGNORECASE)

        # If no matches found, return empty scores
        if not matches:
            logger.warning("No valid scores found in response.")
            return scores

        # Process each match and validate dimensions and scores
        for dim, score in matches:
            dim = dim.strip().lower()
            score = score.strip()

            if dim not in self.dimensions:
                logger.warning(f"Unknown dimension found: {dim}")
                continue

            if not score.isdigit():
                logger.warning(f"Invalid score found for dimension {dim}: {score}")
                continue

            if not (1 <= int(score) <= 5):
                logger.warning(f"Score out of range for dimension {dim}: {score}")
                continue

            scores[dim] = int(score)

        # Check if all dimensions were scored
        if any(score is None for score in scores.values()):
            logger.warning("Some dimensions were not scored in the response.")

        return scores

    def create_requests(self, samples):
        """Prepares Claude batch requests from sample data for each persona."""
        requests = []

        # Create a separate evaluation request for each persona per sample
        for i, sample in enumerate(samples):
            for persona_id, persona_prompt in self.personas.items():
                prompt = self.generate_evaluation_prompt(sample["question"], sample["response"], persona_prompt)

                request = {
                    "custom_id": f"sample_{i}_{persona_id}",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 256,
                        # "temperature": 0.0,
                        # "thinking": {
                        #     "type": "enabled",
                        #     "budget_tokens": 1024  # Small thinking budget for basic analysis
                        # },
                        "messages": [
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    }
                }
                requests.append(request)

        return requests

    def evaluate_batch(self, df, samples, output_path, batch_size=32):
        """Evaluates samples in batches and saves results after each batch."""
        logger.info(f"Saving intermediate results to {output_path}")
        num_samples = len(samples)

        # Process samples in chunks
        for i in tqdm(range(0, num_samples, batch_size), desc="Evaluating with Claude"):
            batch_samples = samples[i:i + batch_size]

            if not batch_samples:
                continue

            requests = self.create_requests(batch_samples)
            batch_id = self._submit_batch(requests)

            # Process the results for the current batch
            batch_results = self._process_batch(batch_id, i)

            if batch_results:
                self.apply_results(df, batch_results)
                self.save_results(df, output_path, i)

        logger.info("All batches processed and final results saved.")

    def save_results(self, df, output_path, batch_start_idx):
        """Safely saves the current DataFrame to disk and logs the result."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved progress for batch starting at index {batch_start_idx}.")
        except Exception as e:
            logger.error(f"Failed to save progress at batch index {batch_start_idx}: {e}")

    def _submit_batch(self, requests):
        """Submits the evaluation batch and waits for processing."""
        batch = self.client.beta.messages.batches.create(requests=requests)
        batch_id = batch.id

        # Poll until processing is done
        while True:
            status = self.client.beta.messages.batches.retrieve(batch_id)
            if status.processing_status == "ended":
                break
            time.sleep(1)

        return batch_id

    def _process_batch(self, batch_id, offset):
        """Extracts and parses Claude's results from the completed batch."""
        results = []
        batch_results = list(self.client.beta.messages.batches.results(batch_id))

        for result in batch_results:
            if result.result.type != "succeeded":
                logger.warning(f"Evaluation failed for sample {result.custom_id}")
                continue

            try:
                # Extract index and persona identifier from custom_id
                _, idx, persona = result.custom_id.split("_", 2)
                idx = int(idx) + offset
            except ValueError:
                logger.warning(f"Invalid custom_id format: {result.custom_id}")
                continue

            # Gather all text and thinking blocks
            text = "".join(block.text for block in result.result.message.content if block.type == "text")
            # thinking = "".join(block.thinking for block in result.result.message.content if block.type == "thinking")
            scores = self.parse_response(text)

            # Extract reasoning directly from the parsed scores, if available
            reasoning = scores.pop("reasoning", "")

            results.append({
                "original_idx": idx,
                "persona": persona,
                "scores": scores,
                "claude/reasoning": reasoning,
                # "claude/thinking": thinking,
                "claude/raw_response": text
            })

        return results

    def apply_results(self, df, results):
        """Applies Claude scores to the dataframe by persona and dimension."""
        for r in results:
            persona = r["persona"]
            idx = r["original_idx"]

            # Assign each score to a persona-prefixed column
            for dim, score in r["scores"].items():
                df.loc[idx, f"claude/{persona}/{dim}"] = score

            # Store raw response and internal model trace (if any)
            df.loc[idx, f"claude/{persona}/reasoning"] = r["claude/reasoning"]
            # df.loc[idx, f"claude/{persona}/thinking"] = r["claude/thinking"]
            df.loc[idx, f"claude/{persona}/raw_response"] = r["claude/raw_response"]
        return df


def load_data(path):
    """Loads CSV and checks required columns."""
    path = Path(path)

    if not path.exists() or path.suffix != ".csv":
        logger.error(f"Invalid input file: {path}")
        return None

    df = pd.read_csv(path)
    required = ["question", "response", "complexity_level"]

    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.error(f"Missing required columns in input file: {missing}")
        return None

    return df


def prepare_samples(df):
    """Prepares records for evaluation."""
    return df[["question", "response", "complexity_level"]].to_dict("records")


def save_dimension_metadata(evaluator, output_csv_path):
    """Saves the dimension and scale metadata to a JSON file."""
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_path = output_path.parent / "dimensions.json"
    
    save_json({"dimensions": evaluator.dimensions, "scale": evaluator.scale}, metadata_path)

    logger.info(f"Saved dimension metadata to {metadata_path}")


def main():
    """Main entry point for evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Evaluate medical QA responses with Claude")
    parser.add_argument("--input-file", required=True, help="CSV input file with generated answers")
    parser.add_argument("--output-file", required=True, help="Where to save output CSV")
    parser.add_argument("--api-key", required=True, help="Anthropic API key for Claude")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for Claude")
    args = parser.parse_args()
    
    df = load_data(args.input_file)
    if df is None:
        return 1
        
    # Filter for specific complexity levels
    target_complexity_levels = [0, 25, 50, 75, 100]
    df_filtered = df[df['complexity_level'].isin(target_complexity_levels)].copy().reset_index(drop=True)
    logger.info(f"Filtered from {len(df)} total samples to {len(df_filtered)} samples with complexity levels {target_complexity_levels}")
    
    evaluator = ClaudeEvaluator(api_key=args.api_key)
    samples = prepare_samples(df_filtered)
    
    logger.info(f"Evaluating {len(samples)} samples...")
    
    # This method now handles applying results and saving the CSV.
    evaluator.evaluate_batch(df_filtered, samples, args.output_file, batch_size=args.batch_size)
    
    # Save the dimension metadata once at the end.
    save_dimension_metadata(evaluator, args.output_file)

    return 0


if __name__ == "__main__":
    main()