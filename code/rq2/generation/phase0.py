import argparse
import logging
import random
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from tqdm import tqdm

import numpy as np
import torch
import pandas as pd
from datasets import load_dataset as hf_load_dataset, Dataset
from lmdeploy import pipeline as turbomind_pipeline, TurbomindEngineConfig, GenerationConfig

from utils.helpers import setup_logging, save_json, load_json

logger = setup_logging(logging.DEBUG)


class ResponseParser:
    """Handles parsing of model responses for XML content."""
    
    # Define all tag names we need to handle
    TAG_NAMES = [
        "complexity_level",
        "answer"
    ]
    
    # Default values as constants
    DEFAULT_COMPLEXITY = None
    DEFAULT_ANSWER = ""
    
    # Compile patterns for each tag
    TAG_PATTERNS = {}
    for tag in TAG_NAMES:
        TAG_PATTERNS[tag] = (
            re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL),  # Standard pattern
            re.compile(f"<{tag}>(.*?)(?=<(?!{tag})|$)", re.DOTALL)  # Fallback pattern
        )
    
    # Pre-compile variant patterns
    VARIANT_PATTERN = re.compile(r'<variant>(.*?)</variant>', re.DOTALL)
    VARIANT_START_PATTERN = re.compile(r'<variant>')
    
    @staticmethod
    def remove_thinking_section(text: str) -> str:
        """Remove thinking trace sections from text."""
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()
        
    @staticmethod
    def extract_content_between_tags(text: str, tag_name: str) -> Optional[str]:
        """Extract content between specific tags with fallback for malformed XML."""
        if not text or tag_name not in ResponseParser.TAG_PATTERNS:
            return None
            
        # Get pre-compiled patterns for this tag
        pattern, fallback_pattern = ResponseParser.TAG_PATTERNS[tag_name]
        
        # Try standard pattern first
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
            
        # Try fallback pattern
        fallback_match = fallback_pattern.search(text)
        if fallback_match:
            logger.warning(f"Using fallback extraction for unclosed <{tag_name}> tag")
            return fallback_match.group(1).strip()
            
        return None

    @staticmethod
    def get_default_response() -> Dict[str, Any]:
        """Return the default response structure."""
        return {
            "complexity_level": ResponseParser.DEFAULT_COMPLEXITY,
            "answer": ResponseParser.DEFAULT_ANSWER
        }

    @staticmethod
    def parse_response(raw_response: str) -> Dict[str, Any]:
        """Parse a single variant from a response."""
        if not raw_response:
            logger.warning("Empty response received")
            return ResponseParser.get_default_response()
            
        # Remove thinking section and work with the full text
        text = ResponseParser.remove_thinking_section(raw_response)
        if not text:
            logger.warning("No content after removing thinking section")
            return ResponseParser.get_default_response()
            
        # Extract the first occurrence of complexity_level and answer
        complexity_level = ResponseParser.extract_content_between_tags(text, "complexity_level")
        answer = ResponseParser.extract_content_between_tags(text, "answer")
        
        # If either tag is missing, return default
        if not complexity_level:
            logger.warning("Missing complexity_level tag in response")
            return ResponseParser.get_default_response()
        if not answer:
            logger.warning("Missing answer tag in response")
            return ResponseParser.get_default_response()
            
        # Try to convert complexity_level to an integer
        try:
            complexity_value = int(complexity_level) if complexity_level.isdigit() else complexity_level
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse complexity level '{complexity_level}': {e}")
            complexity_value = complexity_level
            
        return {
            "complexity_level": complexity_value,
            "answer": answer
        }

    @staticmethod
    def parse_elo_response(raw_response: str) -> List[Dict[str, Any]]:
        """Parse multiple variants from an ELO response."""
        if not raw_response:
            logger.warning("Empty ELO response received")
            return []
            
        # Remove thinking section and work with the full text
        text = ResponseParser.remove_thinking_section(raw_response)
        if not text:
            logger.warning("No content after removing thinking section")
            return []
        
        # Find all starting positions of <variant> tags
        variant_starts = [m.start() for m in ResponseParser.VARIANT_START_PATTERN.finditer(text)]
        if not variant_starts:
            logger.warning("No <variant> tags found in ELO response")
            return []
            
        variants = []
        # Split text into blocks based on <variant> positions
        for i in range(len(variant_starts)):
            start = variant_starts[i]
            end = variant_starts[i + 1] if i + 1 < len(variant_starts) else len(text)
            block = text[start:end]
            
            # Extract complexity_level and answer from each block
            complexity_level = ResponseParser.extract_content_between_tags(block, "complexity_level")
            answer = ResponseParser.extract_content_between_tags(block, "answer")

            if not (complexity_level and answer):
                logger.warning(f"Skipping variant {i+1}: missing complexity_level or answer")
                continue
            
            # Only include variants with both tags present
            try:
                complexity_value = int(complexity_level) if complexity_level.isdigit() else complexity_level
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to parse complexity level '{complexity_level}': {e}")
                complexity_value = complexity_level
                
            variants.append({
                "complexity_level": complexity_value,
                "answer": answer
            })
                
        return variants


class PromptManager:
    """Handles creation of initial prompts."""

    FEW_SHOT_TEMPLATE = """
    You are an expert in creating educational content for different reading abilities. Your task is to rewrite an answer to match a specific complexity level while preserving all factual information.

    When rewriting answers at different complexity levels, you must:
    1. Preserve ALL factual information from the original answer and keep it relevant to the question.
    2. Adjust vocabulary, sentence structure, and explanation detail to match the complexity level.
    3. Do not introduce substantively new claims that aren't reasonably implied by the original answer.
    4. Ensure the answer is coherent and well-structured.
    5. If the original answer does not directly address the question asked, just respond with: '[CONTENT_MISMATCH]' as the answer.

    Complexity levels (1 to 5) are defined as follows:
    - Level 1: For a young child; use very simple vocabulary, short sentences, and basic concepts.
    - Level 2: For a middle school student; use basic scientific terms, clear explanations, and moderate detail.
    - Level 3: For a high school student; use technical terminology, longer sentences, and detailed explanations.
    - Level 4: For a college graduate; use in-depth technical details, complex sentence structures, and scientific language.
    - Level 5: For a biomedical expert; use advanced scientific terminology, assume prior knowledge, and provide precise details.

    Below are examples of how to adjust answers by complexity:

    **Example 1**  
    Question: What causes diabetes?  
    Original Answer: Diabetes is caused by the body's inability to produce or effectively use insulin, a hormone that regulates blood sugar.  
    - Complexity 1: Diabetes happens when your body can't make or use a special helper to control sugar in your blood.
    - Complexity 2: Diabetes happens when your body doesn't make enough insulin or can't use it properly. Insulin is important because it helps control the amount of sugar in your blood.
    - Complexity 3: Diabetes occurs when the pancreas doesn't produce enough insulin or the body can't use insulin properly to manage blood sugar levels.
    - Complexity 4: Diabetes is a metabolic disorder characterized by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both, leading to abnormal carbohydrate metabolism and other biochemical imbalances.
    - Complexity 5: Diabetes results from impaired insulin secretion by pancreatic beta cells or insulin resistance in peripheral tissues, disrupting glucose homeostasis and leading to pathological alterations in protein and lipid metabolism with long-term microvascular and macrovascular complications.

    **Example 2**  
    Question: Why do we get fevers?
    Original Answer: Fevers occur when the body raises its temperature to fight infections, triggered by the immune system's response to pathogens.  
    - Complexity 1: Fevers happen because your body heats up to stop germs from making you sick.
    - Complexity 2: Fevers happen when your body increases its temperature to fight off infections. This helps your immune system work better to kill the germs that are making you sick.
    - Complexity 3: Fevers are caused by the immune system increasing body temperature via the hypothalamus in response to pathogens, enhancing immune function.
    - Complexity 4: Fevers represent a controlled elevation of body temperature orchestrated by the hypothalamus in response to immune system detection of infectious agents. This thermal elevation accelerates metabolic processes, enhances phagocytic activity, and creates a less hospitable environment for pathogens.
    - Complexity 5: Fevers arise from pyrogen-induced activation of the hypothalamic thermoregulatory center, elevating core temperature to augment immune responses against infectious agents. Exogenous and endogenous pyrogens trigger prostaglandin E2 synthesis, which resets the hypothalamic setpoint, initiating physiological responses that elevate and maintain increased core temperature.

    **Example 3**  
    Question: Why is it important to take medications as prescribed?  
    Original Answer: Taking medications as prescribed is essential because it ensures the drug maintains the correct concentration in your body to be effective. Missing doses or incorrect timing can reduce effectiveness or cause side effects.  
    - Complexity 1: It's important to take medicine just like the doctor says. If you don't, the medicine can't help you get better. Sometimes taking medicine wrong can also make you feel bad or sick.
    - Complexity 2: Taking your medicine correctly helps it work the way it should. When you skip doses or take them at the wrong times, there might not be enough medicine in your body to fight the problem, or there might be too much medicine which can cause unwanted side effects.
    - Complexity 3: Following medication prescriptions ensures that drugs maintain therapeutic concentrations in your bloodstream. Deviating from prescribed schedules can result in suboptimal drug levels, reduced efficacy, treatment failure, or adverse reactions due to inconsistent plasma concentrations.
    - Complexity 4: Adherence to medication regimens is critical for maintaining pharmacokinetic profiles within the therapeutic range necessary for clinical efficacy. Erratic dosing disrupts steady-state concentration curves, potentially resulting in periods of subtherapeutic levels associated with treatment failure or toxic concentrations producing adverse effects. Medications with narrow therapeutic indices or nonlinear pharmacokinetics are particularly sensitive to timing irregularities.
    - Complexity 5: Optimal pharmacotherapeutic outcomes depend on maintaining plasma drug concentrations within the therapeutic window—above the minimum effective concentration while below the toxicity threshold. Non-adherence disrupts this homeostasis, potentially resulting in treatment failure or adverse effects. For medications with dose-dependent pharmacokinetics, temporal precision in administration is crucial to maintain the intended therapeutic profile.

    Now, generate an answer variant for the following question and original answer at the specified complexity level: {complexity_level}

    Question: {question}
    Original Answer: {original_answer}
    Complexity Level: {complexity_level}

    Place your response between <root> and </root> tags in exactly this format:
    <root>
      <complexity_level>{complexity_level}</complexity_level>
      <answer>{{Your answer text goes here}}</answer>
    </root>

    Ensure EACH variant has both <complexity_level> and <answer> tags.
    Each tag must be properly closed with the corresponding closing tag.
    Do not include any additional text outside the <root> and </root> tags. Use only the specified XML format.
    """

    ELO_TEMPLATE = """
    You are an expert in creating educational content for different reading abilities. Your task is to generate multiple answer variants for the given question and original answer, each at a specified complexity level, while preserving all factual information.

    When generating each variant, you must:
    1. Preserve ALL factual information from the original answer and keep it relevant to the question.
    2. Adjust vocabulary, sentence structure, and explanation detail to match the complexity level.
    3. Do not introduce substantively new claims that aren't reasonably implied by the original answer.
    4. Ensure the answer is coherent and well-structured.
    5. If the original answer does not directly address the question asked, respond with: '[CONTENT_MISMATCH]' as the answer.

    Complexity levels (1 to 5) are defined as follows:
    - Level 1: For a young child; use very simple vocabulary, short sentences, and basic concepts.
    - Level 2: For a middle school student; use basic scientific terms, clear explanations, and moderate detail.
    - Level 3: For a high school student; use technical terminology, longer sentences, and detailed explanations.
    - Level 4: For a college graduate; use in-depth technical details, complex sentence structures, and scientific language.
    - Level 5: For a biomedical expert; use advanced scientific terminology, assume prior knowledge, and provide precise details.

    Below are examples of how to adjust answers by complexity:

    **Example 1**  
    Question: What causes diabetes?  
    Original Answer: Diabetes is caused by the body's inability to produce or effectively use insulin, a hormone that regulates blood sugar.  
    - Complexity 1: Diabetes happens when your body can't make or use a special helper to control sugar in your blood.
    - Complexity 2: Diabetes happens when your body doesn't make enough insulin or can't use it properly. Insulin is important because it helps control the amount of sugar in your blood.
    - Complexity 3: Diabetes occurs when the pancreas doesn't produce enough insulin or the body can't use insulin properly to manage blood sugar levels.
    - Complexity 4: Diabetes is a metabolic disorder characterized by chronic hyperglycemia resulting from defects in insulin secretion, insulin action, or both, leading to abnormal carbohydrate metabolism and other biochemical imbalances.
    - Complexity 5: Diabetes results from impaired insulin secretion by pancreatic beta cells or insulin resistance in peripheral tissues, disrupting glucose homeostasis and leading to pathological alterations in protein and lipid metabolism with long-term microvascular and macrovascular complications.

    **Example 2**  
    Question: Why do we get fevers?
    Original Answer: Fevers occur when the body raises its temperature to fight infections, triggered by the immune system's response to pathogens.  
    - Complexity 1: Fevers happen because your body heats up to stop germs from making you sick.
    - Complexity 2: Fevers happen when your body increases its temperature to fight off infections. This helps your immune system work better to kill the germs that are making you sick.
    - Complexity 3: Fevers are caused by the immune system increasing body temperature via the hypothalamus in response to pathogens, enhancing immune function.
    - Complexity 4: Fevers represent a controlled elevation of body temperature orchestrated by the hypothalamus in response to immune system detection of infectious agents. This thermal elevation accelerates metabolic processes, enhances phagocytic activity, and creates a less hospitable environment for pathogens.
    - Complexity 5: Fevers arise from pyrogen-induced activation of the hypothalamic thermoregulatory center, elevating core temperature to augment immune responses against infectious agents. Exogenous and endogenous pyrogens trigger prostaglandin E2 synthesis, which resets the hypothalamic setpoint, initiating physiological responses that elevate and maintain increased core temperature.

    **Example 3**  
    Question: Why is it important to take medications as prescribed?  
    Original Answer: Taking medications as prescribed is essential because it ensures the drug maintains the correct concentration in your body to be effective. Missing doses or incorrect timing can reduce effectiveness or cause side effects.  
    - Complexity 1: It's important to take medicine just like the doctor says. If you don't, the medicine can't help you get better. Sometimes taking medicine wrong can also make you feel bad or sick.
    - Complexity 2: Taking your medicine correctly helps it work the way it should. When you skip doses or take them at the wrong times, there might not be enough medicine in your body to fight the problem, or there might be too much medicine which can cause unwanted side effects.
    - Complexity 3: Following medication prescriptions ensures that drugs maintain therapeutic concentrations in your bloodstream. Deviating from prescribed schedules can result in suboptimal drug levels, reduced efficacy, treatment failure, or adverse reactions due to inconsistent plasma concentrations.
    - Complexity 4: Adherence to medication regimens is critical for maintaining pharmacokinetic profiles within the therapeutic range necessary for clinical efficacy. Erratic dosing disrupts steady-state concentration curves, potentially resulting in periods of subtherapeutic levels associated with treatment failure or toxic concentrations producing adverse effects. Medications with narrow therapeutic indices or nonlinear pharmacokinetics are particularly sensitive to timing irregularities.
    - Complexity 5: Optimal pharmacotherapeutic outcomes depend on maintaining plasma drug concentrations within the therapeutic window—above the minimum effective concentration while below the toxicity threshold. Non-adherence disrupts this homeostasis, potentially resulting in treatment failure or adverse effects. For medications with dose-dependent pharmacokinetics, temporal precision in administration is crucial to maintain the intended therapeutic profile.

    Now, generate {num_variants} answer variants for the following question and original answer. The variants should be ordered from the simplest to the most complex, reflecting a gradual increase in complexity. For each variant, assign a complexity level from 1 to 5, where 1 is the simplest and 5 is the most complex, based on the definitions provided.
    
    Question: {question}
    Original Answer: {original_answer}

    Place your response between <root> and </root> tags in exactly this format:
    <root>
      <variant>
        <complexity_level>1</complexity_level>
        <answer>{{Your first answer goes here}}</answer>
      </variant>
      <variant>
        <complexity_level>2</complexity_level>
        <answer>{{Your next answer goes here}}</answer>
      </variant>
      ...
    </root>

    Ensure EACH variant has both <complexity_level> and <answer> tags.
    Each tag must be properly closed with the corresponding closing tag.
    Do not include any additional text outside the <root> and </root> tags. Use only the specified XML format.
    """ 

    @staticmethod
    def create_few_shot_prompt(question: str, original_answer: str, complexity_level: int) -> str:
        """Create a few-shot prompt for a single complexity level."""
        return PromptManager.FEW_SHOT_TEMPLATE.format(
            question=question,
            original_answer=original_answer,
            complexity_level=complexity_level
        )

    @staticmethod
    def create_elo_prompt(question: str, original_answer: str, num_variants: int) -> str:
        """Create an ELO prompt for generating a specified number of variants."""
        return PromptManager.ELO_TEMPLATE.format(
            num_variants=num_variants,
            question=question,
            original_answer=original_answer
        )


class Task:
    """Represents a generation task."""

    def __init__(
        self,
        question_id: str,
        question: str,
        original_answer: str,
        complexity_level: Optional[int] = None,
        num_variants: Optional[int] = None,
        use_elo: bool = False
    ):
        self.question_id = question_id
        self.question = question
        self.original_answer = original_answer
        self.use_elo = use_elo

        if use_elo:
            assert num_variants is not None, "num_variants must be provided for ELO tasks"
            self.num_variants = num_variants
            self.prompt = PromptManager.create_elo_prompt(question, original_answer, num_variants)
        else:
            assert complexity_level is not None, "complexity_level must be provided for few-shot tasks"
            self.complexity_level = complexity_level
            self.prompt = PromptManager.create_few_shot_prompt(question, original_answer, complexity_level)


class TextGenerator:
    """Base class for text generation models."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompts: List[str], temperature: float) -> List[str]:
        raise NotImplementedError("Subclasses must implement generate()")


class TurboMindGenerator(TextGenerator):
    """Generator for TurboMind models with full parameters."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        quant_policy: int = 0,
        cache_max_entry_count: float = 0.7,
        session_len: int = 8192,
        max_new_tokens: int = 4096
    ):
        super().__init__(model_name)

        self.max_new_tokens = max_new_tokens

        engine_config = TurbomindEngineConfig(
            model_format='awq',
            cache_max_entry_count=cache_max_entry_count,
            session_len=session_len,
            tp=1,
            quant_policy=quant_policy,
            enable_prefix_caching=True
        )
        
        self.pipeline = turbomind_pipeline(model_path, backend_config=engine_config)

    def generate(self, prompts: List[str], temperature: float) -> List[str]:
        gen_config = GenerationConfig(max_new_tokens=self.max_new_tokens, temperature=temperature)
        responses = self.pipeline(prompts, gen_config=gen_config)
        return [r.text.strip() for r in responses]


class VariantGenerator:
    """Manages variant generation, including batch processing and result storage."""

    def __init__(
        self,
        generator: TextGenerator,
        temperatures: List[float],
        batch_size: int,
        results: Dict[str, Dict[str, Any]],
        checkpoint_path: str,
        save_every_n_batches: int
    ):
        self.generator = generator
        self.temperatures = temperatures
        self.batch_size = batch_size
        self.results = results
        self.checkpoint_path = checkpoint_path
        self.save_every_n_batches = save_every_n_batches
        self.batch_counter = 0

    def process_tasks(self, tasks: List[Task]) -> None:
        """Process all tasks until completion."""
        for i in range(0, len(tasks), self.batch_size):
            batch = tasks[i:i + self.batch_size]
            self._handle_batch(batch)
            self.batch_counter += 1
            if self.batch_counter % self.save_every_n_batches == 0:
                save_json(self.results, self.checkpoint_path)
                logger.info(f"Checkpoint saved at batch {self.batch_counter}")

    def _handle_batch(self, batch: List[Task]) -> None:
        """Process a batch of tasks."""
        if not batch:
            return
        prompts = [task.prompt for task in batch]
        temperature = random.choice(self.temperatures)
        responses = self.generator.generate(prompts, temperature)

        if len(responses) != len(batch):
            logger.warning(f"Mismatch in batch size: expected {len(batch)}, got {len(responses)}. ")

        for task, response in zip(batch, responses):
            self._process_task(task, response, temperature)

    def _process_task(self, task: Task, raw_response: str, temperature: float) -> None:
        """Process an individual task response."""
        if task.use_elo:
            self._handle_elo_response(task, raw_response, temperature)
        else:
            self._handle_few_shot_response(task, raw_response, temperature)

    def _handle_few_shot_response(self, task: Task, raw_response: str, temperature: float) -> None:
        """Handle a few-shot task response and store it as a single-variant generation."""
        parsed = ResponseParser.parse_response(raw_response)
        variant = {
            "intended_complexity_level": task.complexity_level,
            "complexity_level": parsed["complexity_level"],
            "generated_answer": parsed["answer"]
        }
        self._store_generation(task, raw_response, temperature, [variant])

    def _handle_elo_response(self, task: Task, raw_response: str, temperature: float) -> None:
        """Handle an ELO task response and store multiple variants in one generation."""
        parsed_variants = ResponseParser.parse_elo_response(raw_response)
        if not parsed_variants:
            # If parsing fails, create default variants
            variants = [
                {"complexity_level": None, "generated_answer": ""}
                for _ in range(task.num_variants)
            ]
        else:
            variants = [
                {"complexity_level": variant["complexity_level"], "generated_answer": variant["answer"]}
                for variant in parsed_variants
            ]
        self._store_generation(task, raw_response, temperature, variants)

    def _store_generation(self, task: Task, raw_response: str, temperature: float, variants: List[Dict[str, Any]]) -> None:
        """Store a generation entry containing model info and variants in results."""
        generation = {
            "model_used": self.generator.name,
            "temperature": temperature,
            "raw_response": raw_response,
            "variants": variants
        }
        self.results[task.question_id]["generations"].append(generation)


def load_liveqa_dataset() -> Dataset:
    """Load and process both the LiveQA medical dataset and truehealth/liveqa dataset."""
    # Load the original LiveQA dataset
    liveqa_medical = hf_load_dataset("hyesunyun/liveqa_medical_trec2017", split="test")
    samples = []
    
    # Process hyesunyun/liveqa_medical_trec2017 dataset
    for sample in liveqa_medical:
        question = sample["ORIGINAL_QUESTION_MESSAGE"]
        if len(question.split()) < 10:  
            question = sample["NIST_PARAPHRASE"]
        if len(question.split()) < 10:
            question = sample["NLM_SUMMARY"]
        
        if not question or not question.strip():
            continue

        for answer_obj in sample["REFERENCE_ANSWERS"]:
            answer = answer_obj["ANSWER"]

            if not answer or not answer.strip():
                continue
            
            samples.append({"question": question, "answer": answer})
    
    # Load and process truehealth/liveqa dataset
    truehealth_liveqa = hf_load_dataset("truehealth/liveqa", split="train")
    
    # Process truehealth/liveqa dataset
    for sample in truehealth_liveqa:
        question = sample["message"]
        answer = sample["answer"]

        # Skip if either question or answer is empty
        if not question or not question.strip():
            continue
            
        if not answer or not answer.strip():
            continue
            
        samples.append({"question": question, "answer": answer})
    
    return Dataset.from_list(samples)


def load_medicationqa_dataset() -> Dataset:
    """Load and process the truehealth/medicationqa dataset."""
    medicationqa = hf_load_dataset("truehealth/medicationqa", split="train")
    
    samples = []
    for sample in medicationqa:
        question = sample["Question"]
        answer = sample["Answer"]

        # Skip if either question or answer is empty
        if not question or not question.strip():
            continue
            
        if not answer or not answer.strip():
            continue
            
        samples.append({"question": question, "answer": answer})
    
    return Dataset.from_list(samples)


def load_medquad_dataset() -> Dataset:
    """Load and process the lavita/MedQuAD dataset."""
    medquad = hf_load_dataset("lavita/MedQuAD", split="train")
    
    samples = []
    for sample in medquad:
        question = sample["question"]
        answer = sample["answer"]

        # Skip if either question or answer is empty
        if not question or not question.strip():
            continue
            
        if not answer or not answer.strip():
            continue

        samples.append({"question": question, "answer": answer})
    
    return Dataset.from_list(samples)


def load_bioasq_dataset() -> "Dataset":
    """Load and process the BioASQ dataset."""
    
    # Load the JSON data from the file
    with open("/data/home/djbf/storage/bls/resources/datasets/BioASQ-training13b/training13b.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for question_data in data.get("questions", []):
        question = question_data.get("body", "")
        
        # Skip if question is empty
        if not question or not question.strip():
            continue

        # Get all ideal answers
        ideal_answers = question_data.get("ideal_answer", [])
        if not isinstance(ideal_answers, list):
            ideal_answers = [str(ideal_answers)]

        # Track unique answers to avoid duplicates
        seen_answers = set()
        
        # Create a separate sample for each unique answer
        for answer in ideal_answers:
            # Skip empty answers
            if not answer or not answer.strip():
                continue
                
            # Skip duplicate answers
            if answer in seen_answers:
                continue
            
            # Add this question-answer pair
            samples.append({"question": question, "answer": answer})
            seen_answers.add(answer)
    
    return Dataset.from_list(samples)

def load_mediqaans_dataset() -> Dataset:
    """Load and process the MEDIQA-AnS dataset for summarization tasks."""
    # Load the JSON data from the file
    data_path = "/data/home/djbf/storage/bls/resources/datasets/mediqa-ans/summarization_datasets/question_driven_answer_summarization_primary_dataset.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for question_id, content in data.items():
        question = content.get("question", "")
        
        # Skip if question is empty
        if not question or not question.strip():
            continue
            
        # Get both types of summaries
        multi_abs_summ = content.get("multi_abs_summ", "")
        multi_ext_summ = content.get("multi_ext_summ", "")
        
        # Create a sample with abstractive summary
        if multi_abs_summ and multi_abs_summ.strip():
            samples.append({
                "question": question,
                "answer": multi_abs_summ,
                "summary_type": "abstractive"
            })
            
        # Create a sample with extractive summary
        if multi_ext_summ and multi_ext_summ.strip():
            samples.append({
                "question": question,
                "answer": multi_ext_summ,
                "summary_type": "extractive"
            })
    
    return Dataset.from_list(samples)

def load_custom_dataset(path: str, format: str) -> Dataset:
    """Load a custom dataset from a file."""
    path = Path(path)
    if format.lower() == "csv":
        df = pd.read_csv(path)
    elif format.lower() == "json":
        df = pd.read_json(path)
    else:
        df = pd.read_excel(path)
    return Dataset.from_pandas(df)


def load_dataset(args: argparse.Namespace) -> Dataset:
    """Load the specified dataset."""
    if args.dataset == "liveqa":
        return load_liveqa_dataset()
    elif args.dataset == "medicationqa":
        return load_medicationqa_dataset()
    elif args.dataset == "medquad":
        return load_medquad_dataset()
    elif args.dataset == "bioasq":
        return load_bioasq_dataset()
    elif args.dataset == "mediqaans":
        return load_mediqaans_dataset()
    return load_custom_dataset(args.custom_dataset_path, args.custom_dataset_format)


def initialize_results(dataset: Dataset, checkpoint_path: str) -> Dict[str, Dict[str, Any]]:
    """Initialize or load results structure with unique IDs for each question-answer pair."""
    if Path(checkpoint_path).exists():
        return load_json(checkpoint_path)
    
    results = {}
    seen_pairs = set()
    question_numbers = {}
    answer_counters = defaultdict(int)
    
    for sample in dataset:
        question = sample["question"]
        answer = sample["answer"]
        
        if (question, answer) in seen_pairs:
            logger.info(f"Skipping duplicate question-answer pair: '{question[:30]}...'")
            continue
        
        seen_pairs.add((question, answer))
        
        if question not in question_numbers:
            question_numbers[question] = len(question_numbers) + 1
        
        answer_counters[question] += 1
        key = f"Q{question_numbers[question]}_A{answer_counters[question]}"
        
        results[key] = {
            "question": question,
            "original_answer": answer,
            "generations": []
        }
    
    return results


def get_task_splits(tasks, percentages):
    # Convert to numpy array if it's a list
    tasks = np.array(tasks)

    # Shuffle tasks
    np.random.shuffle(tasks)

    # Calculate split points
    task_count = len(tasks)
    split_indices = [int(p * task_count) for p in np.cumsum(percentages)[:-1]]

    # Split tasks based on percentages
    return np.split(tasks, split_indices)


def get_complexity_levels(num_variants: int) -> List[int]:
    """Generate complexity levels from 1 to 5."""
    if num_variants <= 5:
        # If requesting 5 or fewer variants, return unique evenly distributed levels
        return list(range(1, min(num_variants + 1, 6)))
    
    # If more than 5 variants requested, we'll have to repeat some levels
    # Try to distribute them evenly across the range
    levels = []
    for i in range(num_variants):
        # This ensures even distribution of complexity levels
        level = 1 + (i % 5)
        levels.append(level)
    
    # Shuffle the levels to avoid predictable patterns
    random.shuffle(levels)
    return levels


def generate_tasks(dataset: Dataset, results: Dict[str, Dict[str, Any]], num_variants: int, use_elo: bool) -> List[Task]:
    """Generate tasks for variant creation using the question IDs from results."""
    tasks = []
    
    # Create a mapping from (question, answer) to question_id
    question_answer_to_id = {}
    for question_id, data in results.items():
        question_answer_to_id[(data["question"], data["original_answer"])] = question_id
    
    for sample in dataset:
        question = sample["question"]
        original_answer = sample["answer"]
        
        # Lookup the question_id
        question_id = question_answer_to_id.get((question, original_answer))
        if not question_id:
            logger.warning(f"Could not find question_id for: {question} (answer: {original_answer[:30]}...)")
            continue
        
        if use_elo:
            tasks.append(Task(question_id, question, original_answer, num_variants=num_variants, use_elo=True))
        else:
            # Generate complexity levels for this question-answer pair
            complexity_levels = get_complexity_levels(num_variants)
            for level in complexity_levels:
                tasks.append(Task(question_id, question, original_answer, complexity_level=level))
    
    return tasks


def process_model(
    generator_config: Dict[str, Any],
    tasks: List[Task],
    batch_size: int,
    results: Dict[str, Dict[str, Any]],
    checkpoint_path: str,
    save_every_n_batches: int
) -> None:
    """Process tasks for a single model."""
    generator = TurboMindGenerator(
        model_path=generator_config["path"],
        model_name=generator_config["name"],
        quant_policy=generator_config.get("quant_policy", 0),
        cache_max_entry_count=generator_config.get("cache_max_entry_count", 0.7),
        session_len=generator_config.get("session_len", 8192),
        max_new_tokens=generator_config.get("max_new_tokens", 6144)
    )

    variant_generator = VariantGenerator(
        generator=generator,
        temperatures=generator_config["temperatures"],
        batch_size=batch_size,
        results=results,
        checkpoint_path=checkpoint_path,
        save_every_n_batches=save_every_n_batches
    )
    variant_generator.process_tasks(tasks)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_generator_configs() -> List[Dict[str, Any]]:
    """Load predefined generator configurations."""
    return [
        {
            "path": "/beegfs/client/default/dl-models/turbomind/deepseek-r1-distill-llama-70b-awq-TurboMind",
            "name": "deepseek-llama",
            "quant_policy": 0,
            "cache_max_entry_count": 0.6,
            "session_len": 8192,
            "max_new_tokens": 6144,
            "temperatures": [0.5, 0.7, 0.9, 1.1],
            "percentage": 0.7
        },
        {
            "path": "/beegfs/client/default/dl-models/turbomind/Nvidia-Llama-3.1-Nemotron-70B-Instruct-HF-AWQ-INT4-TurboMind",
            "name": "nvidia-llama",
            "quant_policy": 0,
            "cache_max_entry_count": 0.6,
            "session_len": 8192,
            "max_new_tokens": 6144,
            "temperatures": [0.5, 0.7, 0.9, 1.1],
            "percentage": 0.2
        },
        {
            "path": "/beegfs/client/default/dl-models/turbomind/Llama3-OpenBioLLM-70B-AWQ-INT4-TurboMind",
            "name": "openbiollm-llama",
            "quant_policy": 0,
            "cache_max_entry_count": 0.6,
            "session_len": 8192,
            "max_new_tokens": 6144,
            "temperatures": [0.5, 0.7, 0.9, 1.1],
            "percentage": 0.1
        },
    ]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate answer variants with varying complexity.")
    parser.add_argument("--dataset", type=str, 
                    choices=["liveqa", "medicationqa", "medquad", "bioasq", "mediqaans", "custom"], 
                    default="liveqa", help="Dataset to use")
    parser.add_argument("--custom-dataset-path", type=str, help="Path to custom dataset")
    parser.add_argument("--custom-dataset-format", type=str, default="csv", choices=["csv", "json", "excel", "xlsx"], help="Custom dataset format")
    parser.add_argument("--output", type=str, default="answer_variants.json", help="Output file path")
    parser.add_argument("--num-variants", type=int, default=5, help="Number of variants per question")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of dataset samples")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoint.json", help="Checkpoint file path")
    parser.add_argument("--save-every-n-batches", type=int, default=1, help="Save checkpoint frequency")
    parser.add_argument("--use-elo", action="store_true", help="Use ELO prompt to generate multiple variants per question")
    return parser.parse_args()


def run_variant_generation(args: argparse.Namespace) -> None:
    """Orchestrate the variant generation process."""
    generator_configs = load_generator_configs()
    
    # Filter out models with 0% allocation
    active_configs = [config for config in generator_configs if config["percentage"] > 0]
    
    if not active_configs:
        raise ValueError("No models with non-zero percentage allocation")
        
    dataset = load_dataset(args)
    if args.sample_size and args.sample_size < len(dataset):
        dataset = dataset.select(range(args.sample_size))

    # Initialize results with unique IDs for each question-answer pair
    results = initialize_results(dataset, args.checkpoint_path)
    
    # Generate tasks passing the results dict to access question IDs
    all_tasks = generate_tasks(dataset, results, args.num_variants, args.use_elo)
    
    # Split tasks based on percentages
    percentages = [config["percentage"] for config in active_configs]
    task_splits = get_task_splits(all_tasks, percentages)
    
    # Process each split with its corresponding model
    for config, tasks in zip(active_configs, task_splits):
        tasks = list(tasks)  # Convert from numpy array to list
        logger.info(f"Processing {len(tasks)} tasks with model {config['name']} ({config['percentage']*100:.1f}%)")
        
        if tasks:
            process_model(
                config,
                tasks,
                args.batch_size,
                results,
                args.checkpoint_path,
                args.save_every_n_batches
            )

    # Format the results for output
    final_results = []
    for question_id, data in results.items():
        final_results.append({
            "question_id": question_id,
            "question": data["question"],
            "original_answer": data["original_answer"],
            "generations": data["generations"]
        })
    
    save_json(final_results, args.output)
    logger.info(f"Saved {len(final_results)} question-variant sets to '{args.output}'")


def main():
    """Entry point for variant generation."""
    args = parse_args()
    run_variant_generation(args)


if __name__ == "__main__":
    main()