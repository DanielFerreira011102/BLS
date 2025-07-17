"""
UniEval: A unified evaluator for natural language generation tasks.

This module provides evaluation capabilities for various NLG tasks including:
- Summarization
- Dialogue generation
- Data-to-text generation 
- Factual consistency detection
"""

import sys
from typing import List, Dict, Union, Optional, Any
import numpy as np
from nltk import sent_tokenize
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
from prettytable import PrettyTable


class UniEvaluator:
    """Base evaluator class that handles the model and scoring logic."""
    
    def __init__(self, model_name_or_path: str, max_length: int = 1024, 
                 device: str = 'cuda:0', cache_dir: Optional[str] = None):
        """
        Initialize the evaluator with a pre-trained model.
        
        Args:
            model_name_or_path: HuggingFace model name or path
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache model files
        """
        self.device = device
        self.max_length = max_length
        
        # Load model components
        self.config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, 
            config=self.config,
            cache_dir=cache_dir
        )
        
        # Set up model for evaluation
        self.model.eval()
        self.model.to(device)
        
        # Initialize softmax function and token IDs
        self.softmax = nn.Softmax(dim=1)
        self.pos_id = self.tokenizer("Yes")["input_ids"][0]
        self.neg_id = self.tokenizer("No")["input_ids"][0]
    
    def score(self, inputs: List[str], batch_size: int = 8) -> List[float]:
        """
        Calculate scores for the given inputs.
        
        Args:
            inputs: List of strings to evaluate
            batch_size: Number of samples to process at once
            
        Returns:
            List of scores (between 0 and 1) for each input
        
        Note:
            final_score = positive_score / (positive_score + negative_score)
        """
        # The implementation of "forward" in T5 still requires decoder_input_ids.
        # We construct a dummy target sequence; content doesn't affect scores.
        tgts = ["No" for _ in range(len(inputs))]
        pos_score_list, neg_score_list = [], []
        
        # Process inputs in batches
        for i in tqdm(range(0, len(inputs), batch_size)):
            src_list = inputs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            
            try:
                with torch.no_grad():
                    # Tokenize source and target
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    
                    # Move tensors to device
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)
                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)[:, 0].unsqueeze(-1)
                    
                    # Get model outputs
                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    
                    # Calculate scores
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    pos_score = self.softmax(logits)[:, self.pos_id]  # Yes
                    neg_score = self.softmax(logits)[:, self.neg_id]  # No
                    
                    # Convert to Python list
                    cur_pos_score = [x.item() for x in pos_score]
                    cur_neg_score = [x.item() for x in neg_score]
                    
                    # Store scores
                    pos_score_list.extend(cur_pos_score)
                    neg_score_list.extend(cur_neg_score)
                    
            except RuntimeError as e:
                # Provide debug information
                error_msg = (f"Error processing batch starting at index {i}.\n"
                             f"Source: {src_list}\n"
                             f"Target: {tgt_list}\n"
                             f"Error: {str(e)}")
                raise RuntimeError(error_msg)
        
        # Calculate final scores
        score_list = [
            pos / (pos + neg) 
            for pos, neg in zip(pos_score_list, neg_score_list)
        ]
            
        return score_list


def add_question(dimension: str, output: List[str], 
                 src: Optional[List[str]] = None, 
                 ref: Optional[List[str]] = None, 
                 context: Optional[List[str]] = None, 
                 task: Optional[str] = None) -> List[str]:
    """
    Add questions to generate input in Bool-QA format for UniEval.
    
    Args:
        dimension: Specific dimension to be evaluated
        output: Output text generated by the models
        src: Source input for different NLG tasks
        ref: Human-annotated groundtruth
        context: Additional context needed for evaluation
        task: Task type (summarization, dialogue, data2text, fact)
        
    Returns:
        List of formatted inputs with questions
        
    Raises:
        NotImplementedError: If the dimension or task is not supported
    """
    input_with_question = []
    
    for i in range(len(output)):
        cur_input = ""
        
        # Summarization task
        if task == 'summarization':
            if dimension == 'fluency':
                cur_input = f'question: Is this a fluent paragraph? </s> paragraph: {output[i]}'
            elif dimension == 'coherence':
                cur_input = f'question: Is this a coherent summary to the document? </s> summary: {output[i]} </s> document: {src[i]}'
            elif dimension == 'consistency':
                cur_input = f'question: Is this claim consistent with the document? </s> claim: {output[i]} </s> document: {src[i]}'
            elif dimension == 'relevance':
                cur_input = f'question: Is this summary relevant to the reference? </s> summary: {output[i]} </s> reference: {ref[i]}'
            else:
                raise NotImplementedError(f'Dimension "{dimension}" is not implemented for task "{task}"')
        
        # Dialogue task
        elif task == 'dialogue':
            if dimension == 'naturalness':
                cur_input = f'question: Is this a natural response in the dialogue? </s> response: {output[i]}'
            elif dimension == 'coherence':
                cur_input = (f'question: Is this a coherent response given the dialogue history? </s> response: '
                            f'{output[i]} </s> dialogue history: {src[i]}')
            elif dimension == 'engagingness':
                cur_input = (f'question: Is this an engaging and informative response according to the dialogue history and fact? </s> response: '
                            f'{output[i]} </s> dialogue history: {src[i]} </s> fact: {context[i]}')
            elif dimension == 'groundedness':
                cur_input = (f'question: Is this response consistent with knowledge in the fact? </s> response: '
                            f'{output[i]} </s> fact: {context[i]}')
            elif dimension == 'understandability':
                cur_input = f'question: Is this an understandable response in the dialogue? </s> response: {output[i]}'
            else:
                raise NotImplementedError(f'Dimension "{dimension}" is not implemented for task "{task}"')
        
        # Data-to-text task
        elif task == 'data2text':
            if dimension == 'naturalness':
                cur_input = f'question: Is this a fluent utterance? </s> utterance: {output[i]}'
            elif dimension == 'informativeness':
                cur_input = (f'question: Is this sentence informative according to the reference? </s> sentence: '
                            f'{output[i]} </s> reference: {ref[i]}')
            else:
                raise NotImplementedError(f'Dimension "{dimension}" is not implemented for task "{task}"')
        
        # Factual consistency task
        elif task == 'fact':
            if dimension == 'consistency':
                cur_input = f'question: Is this claim consistent with the document? </s> claim: {output[i]} </s> document: {src[i]}'
            else:
                raise NotImplementedError(f'Only "consistency" dimension is supported for task "{task}"')
        
        # Unsupported task
        else:
            raise NotImplementedError(f'Task "{task}" is not implemented. Please customize specific tasks as needed.')
        
        input_with_question.append(cur_input)
    
    return input_with_question


def convert_to_json(output_list: List[str], 
                   src_list: Optional[List[str]] = None, 
                   ref_list: Optional[List[str]] = None, 
                   context_list: Optional[List[str]] = None,
                   scores: Optional[List[Dict[str, float]]] = None, 
                   doc_id: Optional[List[str]] = None, 
                   system_id: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Convert the data into JSON format.
    
    Args:
        output_list: List of model outputs
        src_list: List of source inputs
        ref_list: List of human-annotated groundtruths
        context_list: List of additional contexts
        scores: List of dictionaries containing human scores
        doc_id: List of document indices
        system_id: List of system indices
        
    Returns:
        List of dictionaries in JSON format
    """
    json_data = []
    
    for i in range(len(output_list)):
        cur = {'system_output': output_list[i]}
        
        if src_list is not None:
            cur['source'] = src_list[i]
        if ref_list is not None:
            cur['reference'] = ref_list[i]
        if context_list is not None:
            cur['context'] = context_list[i]
        if scores is not None:
            cur['scores'] = scores[i]
        if doc_id is not None:
            cur['doc_id'] = doc_id[i]
        if system_id is not None:
            cur['system_id'] = system_id[i]
            
        json_data.append(cur)
        
    return json_data


def print_scores(scores: List[Dict[str, float]]) -> None:
    """
    Print evaluation scores in a formatted table.
    
    Args:
        scores: List of dictionaries containing scores for different dimensions
    """
    table = PrettyTable(['Dimensions', 'Score'])
    print('\nEvaluation scores are shown below:')
    
    dims = list(scores[0].keys())
    for dim in dims:
        # Calculate average score for this dimension
        cur_score = sum(score[dim] for score in scores) / len(scores)
        table.add_row([dim, round(cur_score, 6)])
        
    print(table)


class BaseEvaluator:
    """Base class for all evaluators."""
    
    def __init__(self, max_length: int = 1024, device: str = 'cuda:0', 
                 cache_dir: Optional[str] = None):
        """
        Initialize the base evaluator.
        
        Args:
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache model files
        """
        self.task = None
        self.dimensions = []
        self.scorer = None
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
    
    def _calculate_overall_score(self, eval_scores: List[Dict[str, float]]) -> None:
        """
        Calculate overall score as the average of individual dimension scores.
        
        Args:
            eval_scores: List of dictionaries containing scores
        """
        for i in range(len(eval_scores)):
            eval_scores[i]['overall'] = np.mean(list(eval_scores[i].values()))
    
    def evaluate(self, data: List[Dict[str, str]], 
                dims: Optional[List[str]] = None, 
                overall: bool = True, 
                print_result: bool = False) -> List[Dict[str, float]]:
        """
        Base evaluation method to be implemented by subclasses.
        
        Args:
            data: List of dictionaries containing evaluation data
            dims: List of dimensions to evaluate
            overall: Whether to calculate an overall score
            print_result: Whether to print results
            
        Returns:
            List of dictionaries with evaluation scores
        """
        raise NotImplementedError("Subclasses must implement evaluate()")


class SumEvaluator(BaseEvaluator):
    """Evaluator for text summarization."""
    
    def __init__(self, max_length: int = 1024, device: str = 'cuda:0', 
                 cache_dir: Optional[str] = None):
        """
        Initialize the summarization evaluator.
        
        Args:
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache model files
        """
        super().__init__(max_length, device, cache_dir)
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-sum', 
                                  max_length=max_length, 
                                  device=device, 
                                  cache_dir=cache_dir)
        self.task = 'summarization'
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
    
    def evaluate(self, data: List[Dict[str, str]], 
                dims: Optional[List[str]] = None, 
                overall: bool = True, 
                print_result: bool = False) -> List[Dict[str, float]]:
        """
        Evaluate summarization quality along specified dimensions.
        
        Args:
            data: List of dictionaries containing evaluation data
            dims: List of dimensions to evaluate (defaults to all dimensions)
            overall: Whether to calculate an overall score
            print_result: Whether to print results
            
        Returns:
            List of dictionaries with evaluation scores
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        
        # Determine which dimensions to evaluate
        eval_dims = dims if dims is not None else self.dimensions
        if not isinstance(eval_dims, list):
            raise TypeError("'dims' must be a list or None")
        
        # Evaluate each dimension
        for dim in eval_dims:
            print(f'Evaluating {dim} of {n_data} samples')

            # Handle dimensions that require sentence-level analysis
            if dim in ['consistency', 'fluency']:
                scores = self._evaluate_sentence_level(data, dim)
                
            # Handle dimensions that require summary-level analysis
            elif dim in ['coherence', 'relevance']:
                scores = self._evaluate_summary_level(data, dim)
                
            else:
                raise NotImplementedError(f'Dimension "{dim}" is not implemented for summarization')
            
            # Store scores
            for i in range(n_data):
                eval_scores[i][dim] = scores[i]
        
        # Calculate overall score if requested
        if overall:
            self._calculate_overall_score(eval_scores)
        
        # Print results if requested
        if print_result:
            print_scores(eval_scores)
        
        return eval_scores
    
    def _evaluate_sentence_level(self, data: List[Dict[str, str]], 
                                dim: str) -> List[float]:
        """
        Evaluate dimensions that require sentence-level analysis.
        
        Args:
            data: List of dictionaries containing evaluation data
            dim: Dimension to evaluate ('consistency' or 'fluency')
            
        Returns:
            List of scores for each data point
        """
        src_list, output_list = [], []
        n_sents = []  # Number of sentences in each generated summary
        
        # Prepare data
        for item in data:
            # For consistency, we need the source document
            # For fluency, we use an empty string as source
            source = item['source'] if dim == 'consistency' else ''
            
            # Tokenize output into sentences
            system_outputs = sent_tokenize(item['system_output'])
            n_sents.append(len(system_outputs))
            
            # Add each sentence with its source
            for sentence in system_outputs:
                src_list.append(source)
                output_list.append(sentence)
        
        # Create evaluation input
        input_list = add_question(dimension=dim, output=output_list, 
                                 src=src_list, task=self.task)
        
        # Get scores for each sentence
        sent_score = self.scorer.score(input_list)
        
        # Calculate average score for each summary
        start_idx = 0
        summary_scores = []
        for n_sent in n_sents:
            if n_sent == 0:  # Handle empty summaries
                summary_scores.append(0.0)
                continue
                
            # Calculate average score for this summary
            avg_score = sum(sent_score[start_idx:start_idx + n_sent]) / n_sent
            summary_scores.append(avg_score)
            start_idx += n_sent
        
        return summary_scores
    
    def _evaluate_summary_level(self, data: List[Dict[str, str]], 
                               dim: str) -> List[float]:
        """
        Evaluate dimensions that require summary-level analysis.
        
        Args:
            data: List of dictionaries containing evaluation data
            dim: Dimension to evaluate ('coherence' or 'relevance')
            
        Returns:
            List of scores for each data point
        """
        src_list, output_list, ref_list = [], [], []
        
        # Prepare data
        for item in data:
            src_list.append(item['source'])
            output_list.append(item['system_output'])
            
            # For relevance, we need the reference summary
            if dim == 'relevance':
                ref_list.append(item['reference'])
        
        # Create evaluation input
        input_list = add_question(dimension=dim, output=output_list, 
                                 src=src_list, ref=ref_list, task=self.task)
        
        # Get scores
        scores = self.scorer.score(input_list)
        
        return scores


class DialogEvaluator(BaseEvaluator):
    """Evaluator for dialogues."""
    
    def __init__(self, max_length: int = 1024, device: str = 'cuda:0', 
                 cache_dir: Optional[str] = None):
        """
        Initialize the dialogue evaluator.
        
        Args:
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache model files
        """
        super().__init__(max_length, device, cache_dir)
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-dialog', 
                                  max_length=max_length, 
                                  device=device, 
                                  cache_dir=cache_dir)
        self.task = 'dialogue'
        self.dimensions = ['naturalness', 'coherence', 'engagingness', 
                          'groundedness', 'understandability']

    def evaluate(self, data: List[Dict[str, str]], 
                dims: Optional[List[str]] = None, 
                overall: bool = True, 
                print_result: bool = False) -> List[Dict[str, float]]:
        """
        Evaluate dialogue quality along specified dimensions.
        
        Args:
            data: List of dictionaries containing evaluation data
            dims: List of dimensions to evaluate (defaults to all dimensions)
            overall: Whether to calculate an overall score
            print_result: Whether to print results
            
        Returns:
            List of dictionaries with evaluation scores
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        
        # Determine which dimensions to evaluate
        eval_dims = dims if dims is not None else self.dimensions
        if not isinstance(eval_dims, list):
            raise TypeError("'dims' must be a list or None")
        
        # Evaluate each dimension
        for dim in eval_dims:
            print(f'Evaluating {dim} of {n_data} samples')
            
            # Handle engagingness dimension (sentence-level summation)
            if dim == 'engagingness':
                scores = self._evaluate_engagingness(data)
                
            # Handle other dimensions (turn-level evaluation)
            elif dim in ['naturalness', 'coherence', 'groundedness', 'understandability']:
                scores = self._evaluate_turn_level(data, dim)
                
            else:
                raise NotImplementedError(f'Dimension "{dim}" is not implemented for dialogues')
            
            # Store scores
            for i in range(n_data):
                eval_scores[i][dim] = scores[i]
        
        # Calculate overall score if requested
        if overall:
            self._calculate_overall_score(eval_scores)
        
        # Print results if requested
        if print_result:
            print_scores(eval_scores)
        
        return eval_scores
    
    def _evaluate_engagingness(self, data: List[Dict[str, str]]) -> List[float]:
        """
        Evaluate engagingness dimension (requires sentence-level summation).
        
        Args:
            data: List of dictionaries containing evaluation data
            
        Returns:
            List of scores for each data point
        """
        src_list, output_list, context_list = [], [], []
        n_sents = []  # Number of sentences in each response
        
        # Prepare data
        for item in data:
            source = item['source']
            context = item['context']
            
            # Tokenize output into sentences
            system_outputs = sent_tokenize(item['system_output'])
            n_sents.append(len(system_outputs))
            
            # Add each sentence with its source and context
            for sentence in system_outputs:
                src_list.append(source)
                context_list.append(context)
                output_list.append(sentence)
        
        # Create evaluation input
        input_list = add_question(dimension='engagingness', output=output_list, 
                                 src=src_list, context=context_list, task=self.task)
        
        # Get scores for each sentence
        sent_score = self.scorer.score(input_list)
        
        # Calculate summation score for each response
        start_idx = 0
        response_scores = []
        for n_sent in n_sents:
            if n_sent == 0:  # Handle empty responses
                response_scores.append(0.0)
                continue
                
            # Sum scores for this response
            total_score = sum(sent_score[start_idx:start_idx + n_sent])
            response_scores.append(total_score)
            start_idx += n_sent
        
        return response_scores
    
    def _evaluate_turn_level(self, data: List[Dict[str, str]], 
                            dim: str) -> List[float]:
        """
        Evaluate dimensions that require turn-level analysis.
        
        Args:
            data: List of dictionaries containing evaluation data
            dim: Dimension to evaluate
            
        Returns:
            List of scores for each data point
        """
        src_list, output_list, context_list = [], [], []
        
        # Prepare data
        for item in data:
            # Only coherence needs source
            src_list.append(item['source'] if dim == 'coherence' else '')
            
            output_list.append(item['system_output'])
            
            # Only groundedness needs context
            context_list.append(item['context'] if dim == 'groundedness' else '')
        
        # Create evaluation input
        input_list = add_question(dimension=dim, output=output_list, 
                                 src=src_list, context=context_list, task=self.task)
        
        # Get scores
        scores = self.scorer.score(input_list)
        
        return scores


class D2tEvaluator(BaseEvaluator):
    """Evaluator for data-to-text generation."""
    
    def __init__(self, max_length: int = 1024, device: str = 'cuda:0', 
                 cache_dir: Optional[str] = None):
        """
        Initialize the data-to-text evaluator.
        
        Args:
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache model files
        """
        super().__init__(max_length, device, cache_dir)
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-sum', 
                                  max_length=max_length, 
                                  device=device, 
                                  cache_dir=cache_dir)
        self.task = 'data2text'
        self.dimensions = ['naturalness', 'informativeness']

    def evaluate(self, data: List[Dict[str, str]], 
                dims: Optional[List[str]] = None, 
                overall: bool = True, 
                print_result: bool = False) -> List[Dict[str, float]]:
        """
        Evaluate data-to-text quality along specified dimensions.
        
        Args:
            data: List of dictionaries containing evaluation data
            dims: List of dimensions to evaluate (defaults to all dimensions)
            overall: Whether to calculate an overall score
            print_result: Whether to print results
            
        Returns:
            List of dictionaries with evaluation scores
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        
        # Determine which dimensions to evaluate
        eval_dims = dims if dims is not None else self.dimensions
        if not isinstance(eval_dims, list):
            raise TypeError("'dims' must be a list or None")
        
        # Evaluate each dimension
        for dim in eval_dims:
            print(f'Evaluating {dim} of {n_data} samples')
            
            output_list, ref_list = [], []
            
            # Prepare data
            for item in data:
                output_list.append(item['system_output'])
                ref_list.append(item['reference'])
            
            # Create evaluation input
            input_list = add_question(dimension=dim, output=output_list, 
                                     ref=ref_list, task=self.task)
            
            # Get scores
            scores = self.scorer.score(input_list)
            
            # Store scores
            for i in range(n_data):
                eval_scores[i][dim] = scores[i]
        
        # Calculate overall score if requested
        if overall:
            self._calculate_overall_score(eval_scores)
        
        # Print results if requested
        if print_result:
            print_scores(eval_scores)
        
        return eval_scores


class FactEvaluator(BaseEvaluator):
    """Evaluator for factual consistency detection."""
    
    def __init__(self, max_length: int = 1024, device: str = 'cuda:0', 
                 cache_dir: Optional[str] = None):
        """
        Initialize the factual consistency evaluator.
        
        Args:
            max_length: Maximum sequence length for tokenization
            device: Device to run the model on ('cuda:0', 'cpu', etc.)
            cache_dir: Directory to cache model files
        """
        super().__init__(max_length, device, cache_dir)
        self.scorer = UniEvaluator(model_name_or_path='MingZhong/unieval-fact', 
                                  max_length=max_length, 
                                  device=device, 
                                  cache_dir=cache_dir)
        self.task = 'fact'
        self.dim = 'consistency'
    
    def evaluate(self, data: List[Dict[str, str]], 
                print_result: bool = False) -> List[Dict[str, float]]:
        """
        Evaluate factual consistency.
        
        Args:
            data: List of dictionaries containing evaluation data
            print_result: Whether to print results
            
        Returns:
            List of dictionaries with evaluation scores
        """
        n_data = len(data)
        eval_scores = [{} for _ in range(n_data)]
        
        print(f'Evaluating {self.dim} of {n_data} samples')
        
        # Calculate average sentence-level scores for factual consistency
        src_list, output_list = [], []
        n_sents = []  # Number of sentences in each claim
        
        # Prepare data
        for item in data:
            source = item['source']
            
            # Tokenize output into sentences
            system_outputs = sent_tokenize(item['system_output'])
            n_sents.append(len(system_outputs))
            
            # Add each sentence with its source
            for sentence in system_outputs:
                src_list.append(source)
                output_list.append(sentence)
        
        # Create evaluation input
        input_list = add_question(dimension=self.dim, output=output_list, 
                                 src=src_list, task=self.task)
        
        # Get scores for each sentence
        sent_score = self.scorer.score(input_list)
        
        # Calculate average score for each claim
        start_idx = 0
        claim_scores = []
        for n_sent in n_sents:
            if n_sent == 0:  # Handle empty claims
                claim_scores.append(0.0)
                continue
                
            # Calculate average score for this claim
            avg_score = sum(sent_score[start_idx:start_idx + n_sent]) / n_sent
            claim_scores.append(avg_score)
            start_idx += n_sent
        
        # Store scores
        for i in range(n_data):
            eval_scores[i][self.dim] = claim_scores[i]
        
        # Print results if requested
        if print_result:
            print_scores(eval_scores)
        
        return eval_scores


def get_evaluator(task: str, max_length: int = 1024, device: str = 'cuda:0', 
                 cache_dir: Optional[str] = None) -> BaseEvaluator:
    """
    Get the appropriate evaluator for the specified task.
    
    Args:
        task: Task name ('summarization', 'dialogue', 'data2text', 'fact')
        max_length: Maximum sequence length for tokenization
        device: Device to run the model on ('cuda:0', 'cpu', etc.)
        cache_dir: Directory to cache model files
        
    Returns:
        Evaluator instance for the specified task
        
    Raises:
        ValueError: If the task is not supported
    """
    if task not in ['summarization', 'dialogue', 'data2text', 'fact']:
        raise ValueError(f"Unsupported task: {task}. Supported tasks: "
                        f"'summarization', 'dialogue', 'data2text', 'fact'")
    
    if task == 'summarization':
        return SumEvaluator(max_length=max_length,
                           device=device,
                           cache_dir=cache_dir)
    elif task == 'dialogue':
        return DialogEvaluator(max_length=max_length,
                              device=device,
                              cache_dir=cache_dir)
    elif task == 'data2text':
        return D2tEvaluator(max_length=max_length,
                           device=device,
                           cache_dir=cache_dir)
    elif task == 'fact':
        return FactEvaluator(max_length=max_length,
                            device=device,
                            cache_dir=cache_dir)