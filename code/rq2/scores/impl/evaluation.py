from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import nltk
import torch
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from utils.helpers import setup_logging

logger = setup_logging()


class BaseMetric(ABC):
    """Base abstract class for all text evaluation metrics."""
    
    def __init__(self, name: str):
        # Metric name for identification
        self.name = name
    
    @abstractmethod
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute metric scores for a text pair."""
        pass
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute metrics for a batch of text pairs."""
        # Process each pair individually
        return [self.compute(gen, ref, **kwargs) for gen, ref in zip(generated_texts, reference_texts)]
    
    def __str__(self) -> str:
        """Return metric name as string."""
        return self.name


class RelevanceMetric(BaseMetric):
    """Base class for relevance metrics."""
    
    def __init__(self, name: str):
        super().__init__(name)


class FactualityMetric(BaseMetric):
    """Base class for factuality metrics."""
    
    def __init__(self, name: str):
        super().__init__(name)


class RougeMetric(RelevanceMetric):
    """ROUGE metric for lexical overlap-based relevance."""
    
    def __init__(self, rouge_types: Optional[List[str]] = None, use_stemmer: bool = True):
        super().__init__("ROUGE")
        # Use ROUGE-1, ROUGE-2, ROUGE-L with stemming for standard QA evaluation
        from rouge_score import rouge_scorer
        self.rouge_types = rouge_types or ['rouge1', 'rouge2', 'rougeL']
        self.scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=use_stemmer)
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute ROUGE scores for text pair."""
        # Calculate precision, recall, F1 for each ROUGE type
        scores = self.scorer.score(reference_text, generated_text)
        results = {}
        for rouge_type, score_obj in scores.items():
            results[f"{rouge_type}-p"] = score_obj.precision
            results[f"{rouge_type}-r"] = score_obj.recall
            results[f"{rouge_type}-f1"] = score_obj.fmeasure
        return results


class BleuMetric(RelevanceMetric):
    """BLEU metric for n-gram overlap-based relevance."""
    
    def __init__(self, weights: tuple = (0.25, 0.25, 0.25, 0.25)):
        super().__init__("BLEU")
        # Use NIST smoothing for short QA answers
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        self.sentence_bleu = sentence_bleu
        # Uniform weights for standard comparability
        self.weights = weights
        self.smoothing_function = SmoothingFunction().method3
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute BLEU score for text pair."""
        # Tokenize texts for n-gram comparison
        reference_tokens = [nltk.word_tokenize(reference_text)]
        generated_tokens = nltk.word_tokenize(generated_text)
        bleu_score = self.sentence_bleu(
            reference_tokens, 
            generated_tokens, 
            weights=self.weights,
            smoothing_function=self.smoothing_function
        )
        return {"bleu": bleu_score}


class MeteorMetric(RelevanceMetric):
    """METEOR metric for semantic similarity."""
    
    def __init__(self):
        super().__init__("METEOR")
        # Default settings include synonym matching and stemming
        from nltk.translate.meteor_score import meteor_score
        self.meteor_score = meteor_score
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute METEOR score for text pair."""
        # Tokenize texts for semantic comparison
        reference_tokens = nltk.word_tokenize(reference_text)
        generated_tokens = nltk.word_tokenize(generated_text)
        meteor = self.meteor_score([reference_tokens], generated_tokens)
        return {"meteor": meteor}


class BertScoreMetric(RelevanceMetric):
    """BERTScore metric for semantic similarity."""
    
    def __init__(self, model_type: str = "roberta-large", num_layers: int = 17, 
                 idf: bool = False, device: Optional[str] = None, batch_size: int = 128, lang: str = "en"):
        super().__init__("BERTScore")
        # Use roberta-large, layer 17, no IDF for standard QA evaluation
        self.model_type = model_type
        self.num_layers = num_layers
        self.idf = idf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.lang = lang
        self._scorer = None
    
    def _initialize(self):
        """Initialize BERTScorer with cached model."""
        if self._scorer is not None:
            return

        logger.info(f"Initializing BERTScorer with model {self.model_type}")
        from bert_score import BERTScorer
        self._scorer = BERTScorer(
            model_type=self.model_type,
            num_layers=self.num_layers,
            idf=self.idf,
            device=self.device,
            batch_size=self.batch_size,
            lang=self.lang,
            rescale_with_baseline=False
        )
        logger.info("BERTScorer initialized successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute BERTScore (precision, recall, F1) for text pair."""
        self._initialize()
        P, R, F1 = self._scorer.score([generated_text], [reference_text])
        return {
            "bertscore-p": P.item(),
            "bertscore-r": R.item(),
            "bertscore-f1": F1.item()
        }
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute BERTScore for batch of text pairs."""
        self._initialize()
        P, R, F1 = self._scorer.score(generated_texts, reference_texts)
        return [
            {
                "bertscore-p": p.item(),
                "bertscore-r": r.item(),
                "bertscore-f1": f1.item()
            }
            for p, r, f1 in zip(P, R, F1)
        ]


class SemScoreMetric(RelevanceMetric):
    """SemScore metric for embedding-based similarity."""
    
    def __init__(self, model_path: str = "sentence-transformers/all-mpnet-base-v2", batch_size: int = 128, device: Optional[str] = None):
        super().__init__("SemScore")
        # Use all-mpnet-base-v2 for standard semantic similarity
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model_wrapper = None
    
    def _initialize(self):
        """Initialize sentence transformer model."""
        if self._model_wrapper is not None:
            return

        logger.info(f"Loading SemScore model {self.model_path}")
        from rq2.scores.impl.sem_score import EmbeddingModelWrapper
        self._model_wrapper = EmbeddingModelWrapper(self.model_path, self.batch_size, device=self.device)
        logger.info(f"SemScore model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute cosine similarity for text pair."""
        self._initialize()
        # Get embeddings for both texts
        gen_embeddings = self._model_wrapper.get_embeddings([generated_text])
        ref_embeddings = self._model_wrapper.get_embeddings([reference_text])
        
        # Compute cosine similarity
        similarity = self._model_wrapper.get_similarities(gen_embeddings, ref_embeddings)[0]
        return {"semscore": similarity}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute similarity for batch of text pairs."""
        self._initialize()
        # Get embeddings for all texts at once
        gen_embeddings = self._model_wrapper.get_embeddings(generated_texts)
        ref_embeddings = self._model_wrapper.get_embeddings(reference_texts)
        
        # Compute similarities between corresponding texts
        similarities = self._model_wrapper.get_similarities(gen_embeddings, ref_embeddings)
        return [{"semscore": similarity} for similarity in similarities]


class BleurtMetric(RelevanceMetric):
    """BLEURT metric for semantic similarity."""
    
    def __init__(self, model_name: str = "lucadiliello/BLEURT-20-D12", device: Optional[str] = None, batch_size: int = 128):
        super().__init__("BLEURT")
        # Use BLEURT-20-D12 for efficient, standard QA evaluation
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._config = None
    
    def _initialize(self):
        """Initialize BLEURT model and tokenizer."""
        if self._model is not None:
            return

        logger.info(f"Loading BLEURT model {self.model_name}")
        from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
        self._config = BleurtConfig.from_pretrained(self.model_name)
        self._model = BleurtForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self._tokenizer = BleurtTokenizer.from_pretrained(self.model_name)
        logger.info("BLEURT model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute BLEURT score for text pair."""
        self._initialize()
        self._model.eval()
        with torch.no_grad():
            # Tokenize and score text pair
            inputs = self._tokenizer(
                [reference_text],
                [generated_text],
                padding='longest',
                max_length=512,  # Set maximum sequence length
                truncation=True,  # Truncate sequences longer than max_length
                return_tensors='pt'
            ).to(self.device)
            score = self._model(**inputs).logits.flatten().item()
        return {"bleurt": score}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute BLEURT scores for batch of text pairs."""
        self._initialize()
        self._model.eval()
        with torch.no_grad():
            # Process all pairs in a single batch
            inputs = self._tokenizer(
                reference_texts,
                generated_texts,
                padding='longest',
                max_length=512,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            batch_scores = self._model(**inputs).logits.flatten().tolist()
        
        # Create list of dictionaries with scores
        return [{"bleurt": score} for score in batch_scores]


class BartScoreMetric(RelevanceMetric):
    """BARTScore metric for log-likelihood-based similarity."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: Optional[str] = None, batch_size: int = 128):
        super().__init__("BARTScore")
        # Use bart-large-cnn for standard QA evaluation
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._scorer = None
    
    def _initialize(self):
        """Initialize BARTScore model."""
        if self._scorer is not None:
            return
            
        logger.info(f"Loading BARTScore model {self.model_name}")
        from rq2.scores.impl.bart_score import BARTScorer
        self._scorer = BARTScorer(device=self.device, checkpoint=self.model_name)
        logger.info("BARTScore model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute BARTScore for text pair."""
        self._initialize()
        score = self._scorer.score([generated_text], [reference_text])[0]
        return {"bartscore": score}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute BARTScore for batch of text pairs."""
        self._initialize()
        scores = self._scorer.score(generated_texts, reference_texts)
        return [{"bartscore": score} for score in scores]


class AlignScoreMetric(FactualityMetric):
    """AlignScore metric for NLI-based factuality."""
    
    def __init__(self, model_name: str = "roberta-large", ckpt_path: Optional[str] = "/data/home/djbf/storage/bls/resources/models/AlignScore-large.ckpt", device: Optional[str] = None, batch_size: int = 128):
        super().__init__("AlignScore")
        # model_name is now the base model (roberta-base, roberta-large, etc.)
        self.model_name = model_name 
        # ckpt_path is the path to the checkpoint file
        self.ckpt_path = ckpt_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._aligner = None
    
    def _initialize(self):
        """Initialize AlignScore model."""
        if self._aligner is not None:
            return

        logger.info(f"Loading AlignScore with base model {self.model_name} and checkpoint {self.ckpt_path}")
        from rq2.scores.impl.align_score import AlignScore
        
        self._aligner = AlignScore(
            model=self.model_name,
            batch_size=self.batch_size,
            device=self.device,
            ckpt_path=self.ckpt_path
        )
        logger.info(f"AlignScore model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute AlignScore for text pair."""
        self._initialize()
        score = self._aligner.score(
            contexts=[reference_text],
            claims=[generated_text]
        )[0]
        return {"alignscore": score}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute AlignScore for batch of text pairs."""
        self._initialize()
        scores = self._aligner.score(
            contexts=reference_texts,
            claims=generated_texts
        )
        return [{"alignscore": score} for score in scores]


class SummaCMetric(FactualityMetric):
    """SummaC metric for factual consistency."""
    
    def __init__(self, granularity: str = "sentence", model_name: str = "vitc", device: Optional[str] = None, batch_size: int = 128):
        super().__init__("SummaC")
        # Use vitc model for medical QA factuality, sentence granularity for fine-grained scoring
        self.granularity = granularity
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self._model = None
    
    def _initialize(self):
        """Initialize SummaC model."""
        if self._model is not None:
            return

        logger.info(f"Loading SummaC model {self.model_name}")
        from rq2.scores.impl.summac import SummaCZS
        self._model = SummaCZS(granularity=self.granularity, model_name=self.model_name, device=self.device)
        logger.info("SummaC model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute SummaC score for text pair."""
        self._initialize()
        scores = self._model.score([reference_text], [generated_text])
        return {"summac": scores["scores"][0]}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute SummaC scores for batch of text pairs."""
        self._initialize()
        scores = self._model.score(reference_texts, generated_texts)
        return [{"summac": score} for score in scores["scores"]]

class UniEvalSumMetric(RelevanceMetric):
    """UniEval metric for summarization relevance."""
    
    def __init__(self, max_length: int = 1024, device: Optional[str] = None, 
                 cache_dir: Optional[str] = None, batch_size: int = 8):
        super().__init__("UniEval-Sum")
        # Use MingZhong/unieval-sum for summarization evaluation
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.dimensions = ['coherence', 'consistency', 'fluency', 'relevance']
        self._evaluator = None
    
    def _initialize(self):
        """Initialize UniEval summarization evaluator."""
        if self._evaluator is not None:
            return

        logger.info("Loading UniEval summarization model")
        # Import in function to allow optional dependency
        from rq2.scores.impl.unieval import get_evaluator
        self._evaluator = get_evaluator(
            task='summarization',
            max_length=self.max_length,
            device=self.device,
            cache_dir=self.cache_dir
        )
        logger.info("UniEval summarization model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute UniEval summarization scores for text pair."""
        self._initialize()
        
        # Prepare data for evaluation
        data = [{
            "system_output": generated_text,
            "source": kwargs.get("source", reference_text),
            "reference": reference_text
        }]
        
        # Evaluate using UniEval
        results = self._evaluator.evaluate(data, overall=True, print_result=False)
        
        # Format results
        scores = {}
        for dim, score in results[0].items():
            scores[f"unieval-sum-{dim}"] = score
        
        return scores
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute UniEval summarization scores for batch of text pairs."""
        self._initialize()
        
        # Get source documents if provided, otherwise use reference texts
        sources = kwargs.get("sources", reference_texts)
        
        # Prepare data for evaluation
        data = []
        for i in range(len(generated_texts)):
            data.append({
                "system_output": generated_texts[i],
                "source": sources[i] if i < len(sources) else reference_texts[i],
                "reference": reference_texts[i]
            })
        
        # Evaluate using UniEval
        results = self._evaluator.evaluate(data, overall=True, print_result=False)
        
        # Format results
        scores_list = []
        for result in results:
            scores = {}
            for dim, score in result.items():
                scores[f"unieval-sum-{dim}"] = score
            scores_list.append(scores)
        
        return scores_list


class UniEvalFactMetric(FactualityMetric):
    """UniEval metric for factual consistency detection."""
    
    def __init__(self, max_length: int = 1024, device: Optional[str] = None, 
                 cache_dir: Optional[str] = None, batch_size: int = 8):
        super().__init__("UniEval-Fact")
        # Use MingZhong/unieval-fact for factuality evaluation
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._evaluator = None
    
    def _initialize(self):
        """Initialize UniEval factuality evaluator."""
        if self._evaluator is not None:
            return

        logger.info("Loading UniEval factuality model")
        # Import in function to allow optional dependency
        from rq2.scores.impl.unieval import get_evaluator
        self._evaluator = get_evaluator(
            task='fact',
            max_length=self.max_length,
            device=self.device,
            cache_dir=self.cache_dir
        )
        logger.info("UniEval factuality model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute UniEval factuality score for text pair."""
        self._initialize()
        
        # Prepare data for evaluation
        data = [{
            "system_output": generated_text,
            "source": kwargs.get("source", reference_text)
        }]
        
        # Evaluate using UniEval
        results = self._evaluator.evaluate(data, print_result=False)
        
        # Return the consistency score
        return {"unieval-fact-consistency": results[0]["consistency"]}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute UniEval factuality scores for batch of text pairs."""
        self._initialize()
        
        # Get source documents if provided, otherwise use reference texts
        sources = kwargs.get("sources", reference_texts)
        
        # Prepare data for evaluation
        data = []
        for i in range(len(generated_texts)):
            data.append({
                "system_output": generated_texts[i],
                "source": sources[i] if i < len(sources) else reference_texts[i]
            })
        
        # Evaluate using UniEval
        results = self._evaluator.evaluate(data, print_result=False)
        
        # Format results
        return [{"unieval-fact-consistency": result["consistency"]} for result in results]

class FactCCMetric(FactualityMetric):
    """FactCC metric for factual consistency detection."""
    
    def __init__(self, model_name: str = "manueldeprada/FactCC", device: Optional[str] = None, 
                 batch_size: int = 8, max_length: int = 512):
        super().__init__("FactCC")
        # Use FactCC for factuality evaluation
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        self._factcc = None
    
    def _initialize(self):
        """Initialize FactCC model."""
        if self._factcc is not None:
            return

        logger.info(f"Loading FactCC model {self.model_name}")
        from rq2.scores.impl.factcc import FactCC
        self._factcc = FactCC(
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
            max_length=self.max_length
        )
        logger.info("FactCC model loaded successfully")
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute FactCC score for text pair."""
        self._initialize()
        score = self._factcc.score([reference_text], [generated_text])[0]
        return {"factcc": score}
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute FactCC scores for batch of text pairs."""
        self._initialize()
        scores = self._factcc.score(reference_texts, generated_texts)
        return [{"factcc": score} for score in scores]

class GEvalMetric(BaseMetric):
    """GEval metric for relevance and factuality evaluation."""
    
    def __init__(self, model = None, batch_size: int = 128):
        super().__init__("GEval")
        # Use provided model for GEval evaluation.
        self.batch_size = batch_size
        self._metrics = None
        self._model = model
    
    def _initialize(self):
        """Initialize GEval metrics."""
        # Set up Relevance and Factuality metrics.
        if self._metrics is not None and self._model is not None:
            return

        from rq2.scores.impl.geval import CustomMistral7B
        self._model = CustomMistral7B()
        
        # Configure GEval metrics for evaluation.
        self._metrics = [
            GEval(
                name="Relevance",
                evaluation_steps=[
                    "Check if the 'actual output' directly addresses the 'input' question",
                    "Verify that the 'actual output' includes the key points present in the 'expected output'",
                    "Penalize irrelevant information not related to the 'input' or 'expected output'"
                ],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT
                ],
                model=self._model,
                verbose_mode=True
            ),
            GEval(
                name="Factuality",
                evaluation_steps=[
                    "Check whether the facts in 'actual output' contradict any facts in 'expected output'",
                    "Heavily penalize omission of detail in 'actual output' compared to 'expected output'",
                    "Vague language or contradicting OPINIONS in 'actual output' are acceptable"
                ],
                evaluation_params=[
                    LLMTestCaseParams.INPUT,
                    LLMTestCaseParams.ACTUAL_OUTPUT,
                    LLMTestCaseParams.EXPECTED_OUTPUT
                ],
                model=self._model,
                verbose_mode=True
            )
        ]
    
    def compute(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute GEval scores for text pair."""
        # Initialize metrics if not already set.
        self._initialize()
        
        # Create test case for si   ngle text pair.
        test_case = LLMTestCase(
            input=kwargs.get("input_texts", [""])[0],
            actual_output=generated_text,
            expected_output=reference_text
        )
        
        # Evaluate test case and extract scores.
        results = evaluate(test_cases=[test_case], metrics=self._metrics, max_concurrent=self.batch_size)
        scores = {}
        for metric in results[0].metrics:
            metric_name = metric.name.lower().replace(" ", "-")
            scores[f"geval-{metric_name}"] = metric.score
        
        return scores
    
    def batch_compute(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute GEval scores for batch of text pairs."""
        # Initialize metrics if not already set.
        self._initialize()
        
        # Create test cases for batch of text pairs.
        input_texts = kwargs.get("input_texts", [""] * len(generated_texts))
        test_cases = [
            LLMTestCase(
                input=input_texts[i],
                actual_output=generated_texts[i],
                expected_output=reference_texts[i]
            )
            for i in range(len(generated_texts))
        ]
        
        # Evaluate test cases and extract scores.
        results = evaluate(test_cases=test_cases, metrics=self._metrics, max_concurrent=self.batch_size)
        batch_results = []
        for result in results:
            scores = {}
            for metric in result.metrics:
                metric_name = metric.name.lower().replace(" ", "-")
                scores[f"geval-{metric_name}"] = metric.score
            batch_results.append(scores)
        
        return batch_results


class TextEvaluationMetrics:
    """Unified interface for text evaluation metrics."""
    
    def __init__(self, metrics: Optional[List[BaseMetric]] = None, batch_size: int = 128):
        # Initialize with list of metrics and batch size
        self.metrics = metrics or []
        self.batch_size = batch_size
    
    def add_metric(self, metric: BaseMetric):
        """Add metric to evaluator."""
        self.metrics.append(metric)
    
    def remove_metric(self, metric_name: str):
        """Remove metric by name."""
        self.metrics = [m for m in self.metrics if m.name != metric_name]
    
    def preload_all_models(self):
        """Initialize all models upfront."""
        for metric in self.metrics:
            if hasattr(metric, '_initialize'):
                metric._initialize()
    
    def compute_all(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute all metrics for text pair."""
        results = {}
        for metric in self.metrics:
            results.update(metric.compute(generated_text, reference_text, **kwargs))
        return results
    
    def compute_relevance(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute relevance metrics for text pair."""
        results = {}
        for metric in self.metrics:
            if isinstance(metric, RelevanceMetric):
                results.update(metric.compute(generated_text, reference_text, **kwargs))
        return results
    
    def compute_factuality(self, generated_text: str, reference_text: str, **kwargs) -> Dict[str, float]:
        """Compute factuality metrics for text pair."""
        results = {}
        for metric in self.metrics:
            if isinstance(metric, FactualityMetric):
                results.update(metric.compute(generated_text, reference_text, **kwargs))
        return results
    
    def batch_compute_all(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute all metrics for batch of text pairs."""
        results = [{} for _ in range(len(generated_texts))]
        for metric in self.metrics:
            batch_results = metric.batch_compute(generated_texts, reference_texts, **kwargs)
            for i, result in enumerate(batch_results):
                results[i].update(result)
        return results
    
    def batch_compute_relevance(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute relevance metrics for batch of text pairs."""
        results = [{} for _ in range(len(generated_texts))]
        for metric in self.metrics:
            if isinstance(metric, RelevanceMetric):
                batch_results = metric.batch_compute(generated_texts, reference_texts, **kwargs)
                for i, result in enumerate(batch_results):
                    results[i].update(result)
        return results
    
    def batch_compute_factuality(self, generated_texts: List[str], reference_texts: List[str], **kwargs) -> List[Dict[str, float]]:
        """Compute factuality metrics for batch of text pairs."""
        results = [{} for _ in range(len(generated_texts))]
        for metric in self.metrics:
            if isinstance(metric, FactualityMetric):
                batch_results = metric.batch_compute(generated_texts, reference_texts, **kwargs)
                for i, result in enumerate(batch_results):
                    results[i].update(result)
        return results
    
    def get_available_metrics(self) -> Dict[str, List[str]]:
        """Return available metrics by type."""
        metrics_by_type = {
            "relevance": [],
            "factuality": [],
            "other": []
        }
        for metric in self.metrics:
            if isinstance(metric, RelevanceMetric):
                metrics_by_type["relevance"].append(str(metric))
            elif isinstance(metric, FactualityMetric):
                metrics_by_type["factuality"].append(str(metric))
            else:
                metrics_by_type["other"].append(str(metric))
        return metrics_by_type


def main():
    """Demonstrate usage with medical QA examples."""    
    text_pairs = [
        {
            "input": "What is hypertension and what are its risks?",
            "reference": "Hypertension is a chronic medical condition characterized by elevated blood pressure above 140/90 mmHg. It increases the risk of heart disease and stroke.",
            "generated": "High blood pressure, or hypertension, is a long-term condition where blood pressure exceeds 140/90 mmHg, raising risks for cardiovascular issues."
        },
        {
            "input": "How is Type 2 diabetes managed?",
            "reference": "Type 2 diabetes is managed through lifestyle changes, metformin, and insulin therapy in advanced cases. Regular monitoring of blood glucose is essential.",
            "generated": "Type 2 diabetes treatment includes diet, exercise, metformin, and sometimes insulin. Blood sugar levels should be checked regularly."
        },
        {
            "input": "Are antibiotics effective against viral infections?",
            "reference": "Antibiotics are ineffective against viral infections like the common cold. Overuse can lead to antibiotic resistance, a major public health concern.",
            "generated": "Antibiotics don't work for viruses such as the cold and their overuse contributes to resistance, posing a significant health risk."
        }
    ]
    
    # Initialize evaluator with all metrics including GEvalMetric
    evaluator = TextEvaluationMetrics(metrics=[
        RougeMetric(),
        BleuMetric(weights=(0.25, 0.25, 0.25, 0.25)),
        MeteorMetric(),
        BertScoreMetric(),
        SemScoreMetric(),
        BleurtMetric(),
        BartScoreMetric(),
        AlignScoreMetric(),
    ])
    
    evaluator.preload_all_models()
    
    # Extract data from text_pairs
    references = [pair["reference"] for pair in text_pairs]
    generated = [pair["generated"] for pair in text_pairs]
    inputs = [pair["input"] for pair in text_pairs]
    
    # Compute metrics with input_texts
    batch_results = evaluator.batch_compute_all(generated, references, input_texts=inputs)
    
    # Print results for each text pair
    for idx, (pair, results) in enumerate(zip(text_pairs, batch_results), 1):
        print(f"\n=== Text Pair {idx} ===")
        print(f"Input: {pair['input']}")
        print(f"Reference: {pair['reference']}")
        print(f"Generated: {pair['generated']}")
        print("Metrics:")
        for metric_name, score in results.items():
            print(f"  {metric_name}: {score:.4f}")
    
    metrics = evaluator.get_available_metrics()
    print(f"\nRelevance metrics: {', '.join(metrics['relevance'])}")
    print(f"Factuality metrics: {', '.join(metrics['factuality'])}")
    print(f"Other metrics: {', '.join(metrics['other'])}")


if __name__ == "__main__":
    main()