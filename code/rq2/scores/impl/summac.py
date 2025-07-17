"""
Simplified SummaC implementation for factual consistency evaluation.
"""

import os
import json
import nltk
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# NLI model mapping
MODEL_MAP = {
    "snli-base": {"model_card": "boychaboy/SNLI_roberta-base", "entailment_idx": 0, "contradiction_idx": 2},
    "snli-large": {"model_card": "boychaboy/SNLI_roberta-large", "entailment_idx": 0, "contradiction_idx": 2},
    "mnli-base": {"model_card": "microsoft/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli": {"model_card": "roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {"model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", "entailment_idx": 0, "contradiction_idx": 2},
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc": {"model_card": "tals/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-only": {"model_card": "tals/albert-xlarge-vitaminc", "entailment_idx": 0, "contradiction_idx": 1},
}


def name_to_card(name: str) -> str:
    """Convert model name to model card."""
    if name not in MODEL_MAP:
        return name
    return MODEL_MAP[name]["model_card"]


def get_neutral_idx(ent_idx: int, con_idx: int) -> int:
    """Get the index for neutral class."""
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]


def batcher(data: List[Any], batch_size: int) -> List[List[Any]]:
    """Split data into batches."""
    n = len(data)
    return [data[i:min(i + batch_size, n)] for i in range(0, n, batch_size)]


class SummaCImager:
    """Create NLI-based images for comparing original and generated text."""
    
    def __init__(self, model_name: str = "mnli", granularity: str = "paragraph", 
                 use_cache: bool = True, max_doc_sents: int = 100, 
                 device: str = "cuda", **kwargs):
        """Initialize the SummaCImager."""
        self.grans = granularity.split("-")
        
        if not all(gran in ["paragraph", "sentence", "document", "2sents", "mixed"] for gran in self.grans) or len(self.grans) > 2:
            raise ValueError(f"Unrecognized granularity {granularity}")
        
        if model_name not in MODEL_MAP:
            raise ValueError(f"Unrecognized model name: {model_name}")

        self.model_name = model_name
        self.model_card = name_to_card(model_name)
        self.entailment_idx = MODEL_MAP[model_name]["entailment_idx"]
        self.contradiction_idx = MODEL_MAP[model_name]["contradiction_idx"]
        self.neutral_idx = get_neutral_idx(self.entailment_idx, self.contradiction_idx)

        self.granularity = granularity
        self.use_cache = use_cache
        self.cache_folder = os.path.join(os.path.expanduser("~"), ".summac_cache")
        os.makedirs(self.cache_folder, exist_ok=True)

        self.max_doc_sents = max_doc_sents
        self.max_input_length = 500
        self.device = device
        self.cache = {}
        self.model = None
        self.tokenizer = None

    def load_nli(self) -> None:
        """Load the NLI model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_card).eval()
        self.model.to(self.device)
        if self.device == "cuda":
            self.model.half()

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = nltk.tokenize.sent_tokenize(text)
        return [sent for sent in sentences if len(sent) > 10]

    def split_2sents(self, text: str) -> List[str]:
        """Split text into chunks of 2 sentences each."""
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return [" ".join(sentences[i:(i+2)]) for i in range(len(sentences))]

    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        if text.count("\n\n") > 0:
            paragraphs = [p.strip() for p in text.split("\n\n")]
        else:
            paragraphs = [p.strip() for p in text.split("\n")]
        return [p for p in paragraphs if len(p) > 10]

    def split_text(self, text: str, granularity: str = "sentence") -> List[str]:
        """Split text according to the specified granularity."""
        if granularity == "document":
            return [text]
        if granularity == "paragraph":
            return self.split_paragraphs(text)
        if granularity == "sentence":
            return self.split_sentences(text)
        if granularity == "2sents":
            return self.split_2sents(text)
        if granularity == "mixed":
            return self.split_sentences(text) + self.split_paragraphs(text)
        return [text]

    def build_chunk_dataset(self, original: str, generated: str, pair_idx: Optional[int] = None) -> Tuple[List[Dict], int, int]:
        """Build a dataset of chunks for comparison."""
        if len(self.grans) == 1:
            gran_doc, gran_sum = self.grans[0], self.grans[0]
        else:
            gran_doc, gran_sum = self.grans[0], self.grans[1]

        original_chunks = self.split_text(original, granularity=gran_doc)[:self.max_doc_sents]
        generated_chunks = self.split_text(generated, granularity=gran_sum)

        N_ori, N_gen = len(original_chunks), len(generated_chunks)
        dataset = []
        
        for i in range(N_ori):
            for j in range(N_gen):
                dataset.append({
                    "premise": original_chunks[i], 
                    "hypothesis": generated_chunks[j], 
                    "doc_i": i, 
                    "gen_i": j, 
                    "pair_idx": pair_idx
                })
        
        return dataset, N_ori, N_gen

    def build_image(self, original: str, generated: str) -> np.ndarray:
        """Build an NLI image for original and generated text pair."""
        cache_key = (original, generated)
        if self.use_cache and cache_key in self.cache:
            cached_image = self.cache[cache_key]
            return cached_image[:, :self.max_doc_sents, :]

        dataset, N_ori, N_gen = self.build_chunk_dataset(original, generated)
        if len(dataset) == 0:
            return np.zeros((3, 1, 1))

        image = np.zeros((3, N_ori, N_gen))
        if self.model is None:
            self.load_nli()

        for batch in batcher(dataset, batch_size=20):
            batch_prems = [b["premise"] for b in batch]
            batch_hypos = [b["hypothesis"] for b in batch]
            batch_tokens = self.tokenizer.batch_encode_plus(
                list(zip(batch_prems, batch_hypos)), 
                padding=True, 
                truncation=True, 
                max_length=self.max_input_length, 
                return_tensors="pt", 
                truncation_strategy="only_first"
            )
            
            with torch.no_grad():
                batch_inputs = {k: v.to(self.device) for k, v in batch_tokens.items()}
                model_outputs = self.model(**batch_inputs)

            batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
            batch_evids = batch_probs[:, self.entailment_idx].tolist()
            batch_conts = batch_probs[:, self.contradiction_idx].tolist()
            batch_neuts = batch_probs[:, self.neutral_idx].tolist()

            for b, evid, cont, neut in zip(batch, batch_evids, batch_conts, batch_neuts):
                image[0, b["doc_i"], b["gen_i"]] = evid
                image[1, b["doc_i"], b["gen_i"]] = cont
                image[2, b["doc_i"], b["gen_i"]] = neut

        if self.use_cache:
            self.cache[cache_key] = image
            
        return image

    def build_images(self, originals: List[str], generateds: List[str], batch_size: int = 128) -> List[np.ndarray]:
        """Build images for multiple text pairs."""
        # Find which pairs need processing
        todo_originals, todo_generateds = [], []
        for ori, gen in zip(originals, generateds):
            if (ori, gen) not in self.cache:
                todo_originals.append(ori)
                todo_generateds.append(gen)
        
        # Nothing to process, return from cache
        if not todo_originals:
            return [self.cache.get((ori, gen)) for ori, gen in zip(originals, generateds)]
        
        # Process new pairs
        total_dataset = []
        todo_images = []
        
        for pair_idx, (ori, gen) in enumerate(zip(todo_originals, todo_generateds)):
            dataset, N_ori, N_gen = self.build_chunk_dataset(ori, gen, pair_idx=pair_idx)
            image = np.zeros((3, N_ori if dataset else 1, N_gen if dataset else 1))
            todo_images.append(image)
            total_dataset.extend(dataset)
            
        if self.model is None:
            self.load_nli()
        
        # Process batches
        for batch in batcher(total_dataset, batch_size=batch_size):
            batch_prems = [b["premise"] for b in batch]
            batch_hypos = [b["hypothesis"] for b in batch]
            batch_tokens = self.tokenizer.batch_encode_plus(
                list(zip(batch_prems, batch_hypos)), 
                padding=True, 
                truncation=True, 
                max_length=self.max_input_length, 
                return_tensors="pt", 
                truncation_strategy="only_first"
            )
            
            with torch.no_grad():
                batch_inputs = {k: v.to(self.device) for k, v in batch_tokens.items()}
                model_outputs = self.model(**batch_inputs)

            batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
            batch_evids = batch_probs[:, self.entailment_idx].tolist()
            batch_conts = batch_probs[:, self.contradiction_idx].tolist()
            batch_neuts = batch_probs[:, self.neutral_idx].tolist()

            for b, evid, cont, neut in zip(batch, batch_evids, batch_conts, batch_neuts):
                image = todo_images[b["pair_idx"]]
                image[0, b["doc_i"], b["gen_i"]] = evid
                image[1, b["doc_i"], b["gen_i"]] = cont
                image[2, b["doc_i"], b["gen_i"]] = neut

        # Update cache
        for pair_idx, (ori, gen) in enumerate(zip(todo_originals, todo_generateds)):
            self.cache[(ori, gen)] = todo_images[pair_idx]

        # Return all images
        return [self.cache.get((ori, gen)) for ori, gen in zip(originals, generateds)]

    def get_cache_file(self) -> str:
        """Get the path to the cache file."""
        return os.path.join(self.cache_folder, f"cache_{self.model_name}_{self.granularity}.json")

    def save_cache(self) -> None:
        """Save the cache to disk."""
        cache_cp = {"[///]".join(k): v.tolist() for k, v in self.cache.items()}
        with open(self.get_cache_file(), "w") as f:
            json.dump(cache_cp, f)

    def load_cache(self) -> None:
        """Load the cache from disk."""
        cache_file = self.get_cache_file()
        if not os.path.isfile(cache_file):
            return
            
        with open(cache_file, "r") as f:
            cache_cp = json.load(f)
            self.cache = {tuple(k.split("[///]")): np.array(v) for k, v in cache_cp.items()}


class SummaCZS:
    """Zero-shot approach for measuring factual consistency in summarization."""
    
    def __init__(self, model_name: str = "mnli", granularity: str = "paragraph", 
                 op1: str = "max", op2: str = "mean", use_ent: bool = True, 
                 use_con: bool = True, imager_load_cache: bool = True, 
                 device: str = "cuda", **kwargs):
        """Initialize the SummaCZS model."""
        if op2 not in ["min", "mean", "max"]:
            raise ValueError(f"Unrecognized op2: {op2}")
            
        if op1 not in ["max", "mean", "min"]:
            raise ValueError(f"Unrecognized op1: {op1}")
        
        self.device = device
        self.imager = SummaCImager(
            model_name=model_name, 
            granularity=granularity, 
            device=self.device, 
            **kwargs
        )
        
        if imager_load_cache:
            self.imager.load_cache()
            
        self.op2 = op2
        self.op1 = op1
        self.use_ent = use_ent
        self.use_con = use_con

    def save_imager_cache(self) -> None:
        """Save the imager's cache to disk."""
        self.imager.save_cache()

    def score_one(self, original: str, generated: str) -> Dict[str, Union[np.ndarray, float]]:
        """Score a single document-summary pair."""
        image = self.imager.build_image(original, generated)
        score = self.image2score(image)
        return {"image": image, "score": score}

    def image2score(self, image: np.ndarray) -> float:
        """Convert an NLI image to a factual consistency score."""
        # First aggregation over document chunks
        if self.op1 == "max":
            ent_scores = np.max(image[0], axis=0)
            co_scores = np.max(image[1], axis=0)
        elif self.op1 == "mean":
            ent_scores = np.mean(image[0], axis=0)
            co_scores = np.mean(image[1], axis=0)
        else:  # "min"
            ent_scores = np.min(image[0], axis=0)
            co_scores = np.min(image[1], axis=0)

        # Combine entailment and contradiction scores
        if self.use_ent and self.use_con:
            scores = ent_scores - co_scores
        elif self.use_ent:
            scores = ent_scores
        else:  # use_con only
            scores = 1.0 - co_scores

        # Second aggregation for final score
        if self.op2 == "min":
            final_score = np.min(scores)
        elif self.op2 == "mean":
            final_score = np.mean(scores)
        else:  # "max"
            final_score = np.max(scores)
            
        return final_score

    def score(self, sources: List[str], generateds: List[str], 
              batch_size: int = 128) -> Dict[str, Union[List[float], List[np.ndarray]]]:
        """Score multiple document-summary pairs."""
        images = self.imager.build_images(sources, generateds, batch_size=batch_size)
        scores = [self.image2score(image) for image in images]
        return {"scores": scores, "images": images}


if __name__ == "__main__":
    # Example usage
    model = SummaCZS(
        granularity="document", 
        model_name="vitc", 
        imager_load_cache=True, 
        device="cpu"  # Use "cuda" if available
    )

    document = "Jeff joined Microsoft in 1992 to lead corporate developer evangelism for Windows NT."
    summary1 = "Jeff joined Microsoft in 1992."
    summary2 = "Jeff joined Microsoft."

    # Score the summaries
    results = model.score([document, document], [summary1, summary2])
    print(f"Scores: {results['scores']}")