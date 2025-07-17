import numpy as np
from typing import List, Tuple, Optional

import nltk
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

class MaskedProbabilityClassifier:
    """
    Implementation of the masked probability metric from the paper
    "Paragraph-level Simplification of Medical Texts" and "Readability Controllable Biomedical Document Summarization"
    """
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", seed: int = 42, 
                 num_masks: int = 10, mask_fraction: float = 0.15, method: str = "random"):
        """Initialize the metric with a pre-trained masked language model."""        
        # Store masking parameters
        self.default_num_masks = num_masks
        self.default_mask_fraction = mask_fraction 
        self.default_method = method
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Set model to evaluation mode
        self.model.eval()
        
        # Get the mask token ID
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # Get CLS and SEP token IDs to avoid masking them
        self.special_tokens = [
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id
        ]

        # Set random seed
        np.random.seed(seed)
        
        # Lazy loading for spaCy
        self.nlp = None

    def _load_spacy_if_needed(self) -> bool:
        """Lazy load spaCy model when needed."""
        if self.nlp is not None:
            return True
            
        import spacy
        spacy.prefer_gpu()  # Use GPU if available

        self.nlp = spacy.load("en_core_web_sm")
        return True

    def predict_single(self, text, num_masks=None, mask_fraction=None, method=None):
        """Compute complexity score for a single text."""
        num_masks = num_masks if num_masks is not None else self.default_num_masks
        mask_fraction = mask_fraction if mask_fraction is not None else self.default_mask_fraction
        method = method if method is not None else self.default_method
        
        # Use random token masking
        if method == "random":
            return self.masked_prob(text, num_masks, mask_fraction)
            
        # NP masking methods require spaCy
        if not self._load_spacy_if_needed():
            print("Error: spaCy model not loaded. Falling back to random masking.")
            return self.masked_prob(text, num_masks, mask_fraction)
        
        # Use RNPTC method (best according to paper)
        if method == "rnptc":
            return self.ranked_np_masked_prob(text)
            
        # Use basic NP masking
        if method == "np":
            return self.np_masked_prob(text, num_masks)
            
        # Default to random masking for unknown methods
        print(f"Unknown method '{method}'. Falling back to random masking.")
        return self.masked_prob(text, num_masks, mask_fraction)
            
    def predict_batch(self, texts, num_masks=None, mask_fraction=None, method=None):
        """
        Compute scores for a batch of texts.
        Note: We process each text individually as true batch processing
        would require significant restructuring of the masking logic.
        """
        num_masks = num_masks if num_masks is not None else self.default_num_masks
        mask_fraction = mask_fraction if mask_fraction is not None else self.default_mask_fraction
        method = method if method is not None else self.default_method
        
        return [self.predict_single(text, num_masks, mask_fraction, method) for text in texts]

    def masked_prob(self, text: str, num_masks: int = 30, mask_fraction: float = 0.15) -> float:
        """Compute the masked probability score for a text document."""
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        all_probs = []
        
        # Process each sentence
        for sent in sentences:
            # Tokenize the sentence
            tokens = self.tokenizer.encode(sent, return_tensors="pt").to(self.device)            

            # Skip sentences that are too short
            if tokens.size(1) <= 3:  # Just CLS, one token, and SEP
                continue
                
            # Get indices of tokens that can be masked (exclude special tokens)
            maskable_indices = [
                i for i in range(1, tokens.size(1) - 1)  # Skip CLS and SEP
                if tokens[0, i].item() not in self.special_tokens
            ]
            
            # Skip if no maskable tokens
            if not maskable_indices:
                continue
                
            # Calculate number of tokens to mask
            num_to_mask = max(1, int(len(maskable_indices) * mask_fraction))
            
            # Process multiple masking patterns
            for _ in range(num_masks):
                # Sample indices to mask
                if len(maskable_indices) <= num_to_mask:
                    indices_to_mask = maskable_indices
                else:
                    indices_to_mask = np.random.choice(
                        maskable_indices, 
                        size=num_to_mask, 
                        replace=False
                    ).tolist()
                
                # Create a copy of the tokens and apply masking
                masked_tokens = tokens.clone()
                for idx in indices_to_mask:
                    masked_tokens[0, idx] = self.mask_token_id
                
                # Forward pass through model
                with torch.no_grad():
                    outputs = self.model(masked_tokens)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                # Extract probabilities of original tokens at masked positions
                for idx in indices_to_mask:
                    original_token_id = tokens[0, idx].item()
                    prob = probs[0, idx, original_token_id].item()
                    all_probs.append(prob)
        
        # Return mean probability if we have any results
        return sum(all_probs) / len(all_probs) if all_probs else 0.0
    
    def extract_noun_phrases(self, text: str) -> List[Tuple[str, int, int]]:
        """Extract noun phrases (NPs) from text using spaCy."""
        if not self._load_spacy_if_needed():
            return []
            
        doc = self.nlp(text)
        noun_phrases = []
        
        # Extract all noun phrases
        for chunk in doc.noun_chunks:
            # Filter out NPs that consist only of stopwords
            if all(token.is_stop for token in chunk):
                continue
                
            # Record the span and text
            noun_phrases.append((chunk.text, chunk.start_char, chunk.end_char))
                
        return noun_phrases
    
    def np_masked_prob(self, text: str, num_masks: int = 10) -> float:
        """
        Compute the masked probability score using noun phrase masking.
        This follows the approach described in the paper where noun phrases
        are masked to better evaluate biomedical text complexity.
        """
        if not self._load_spacy_if_needed():
            raise ValueError("spaCy model not loaded. Cannot use NP masking.")
            
        # Extract noun phrases
        noun_phrases = self.extract_noun_phrases(text)
        
        # Skip if no NPs found
        if not noun_phrases:
            return 0.0
            
        all_probs = []
        
        # Tokenize the full text
        original_tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # For each trial, randomly select an NP to mask
        for _ in range(num_masks):
            if not noun_phrases:
                continue
                
            # Randomly select an NP
            idx = np.random.choice(len(noun_phrases))
            np_text, start_char, end_char = noun_phrases[idx]
            
            # Get the text before the NP to find the starting position
            prefix_text = text[:start_char]
            
            # Find position of the NP tokens
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=True)
            original_np_tokens = self.tokenizer.encode(np_text, add_special_tokens=False)
            start_pos = len(prefix_tokens) - 1  # Subtracting 1 for the CLS token
            end_pos = start_pos + len(original_np_tokens)
            
            # Check if NP position is within bounds of the encoded text
            if end_pos >= original_tokens.size(1):
                continue
            
            # Create masked tokens by replacing the NP tokens with mask tokens
            masked_tokens = original_tokens.clone()
            for pos in range(start_pos, end_pos):
                if pos < masked_tokens.size(1):  # Safety check
                    masked_tokens[0, pos] = self.mask_token_id
            
            # Forward pass through model
            with torch.no_grad():
                outputs = self.model(masked_tokens)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Calculate average probability for the NP
            np_prob = 0.0
            count = 0
            for i, pos in enumerate(range(start_pos, end_pos)):
                if i >= len(original_np_tokens) or pos >= original_tokens.size(1):
                    continue
                original_token_id = original_np_tokens[i]
                prob = probs[0, pos, original_token_id].item()
                np_prob += prob
                count += 1
            
            if count > 0:
                all_probs.append(np_prob / count)
        
        # Return mean probability if we have any results
        return sum(all_probs) / len(all_probs) if all_probs else 0.0
        
    def ranked_np_masked_prob(self, text: str) -> float:
        """
        Implementation of the RNPTC (Ranked NP-based Text Complexity) metric.
        This assigns weights to token probabilities depending on their ranking.
        """
        if not self._load_spacy_if_needed():
            raise ValueError("spaCy model not loaded. Cannot use RNPTC.")
            
        # Extract all noun phrases
        noun_phrases = self.extract_noun_phrases(text)
        
        # Skip if no NPs found
        if not noun_phrases:
            return 0.0
            
        np_probs = []
        
        # Tokenize the full text
        original_tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Calculate probability for each NP
        for np_text, start_char, end_char in noun_phrases:
            # Get prefix text and tokenize
            prefix_text = text[:start_char]
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=True)
            original_np_tokens = self.tokenizer.encode(np_text, add_special_tokens=False)
            
            # Find position of the NP tokens
            start_pos = len(prefix_tokens) - 1
            end_pos = start_pos + len(original_np_tokens)
            
            # Check if NP position is within bounds of the encoded text
            if end_pos >= original_tokens.size(1):
                continue
            
            # Create masked tokens
            masked_tokens = original_tokens.clone()
            for pos in range(start_pos, end_pos):
                if pos < masked_tokens.size(1):
                    masked_tokens[0, pos] = self.mask_token_id
            
            # Forward pass through model
            with torch.no_grad():
                outputs = self.model(masked_tokens)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Calculate average probability for the NP
            np_prob = 0.0
            count = 0
            for i, pos in enumerate(range(start_pos, end_pos)):
                if i >= len(original_np_tokens) or pos >= original_tokens.size(1):
                    continue
                original_token_id = original_np_tokens[i]
                prob = probs[0, pos, original_token_id].item()
                np_prob += prob
                count += 1
            
            if count > 0:
                np_probs.append(np_prob / count)
        
        # Sort NP probabilities in descending order
        np_probs.sort(reverse=True)
        
        if not np_probs:
            return 0.0
            
        # Calculate ranked mean
        ranked_sum = 0.0
        for i, prob in enumerate(np_probs, 1):
            ranked_sum += prob / np.sqrt(i)
            
        score = ranked_sum / len(np_probs)
        
        # Return the negative log
        if score <= 0:
            return 0.0
            
        return -np.log(score)

if __name__ == "__main__":
    # Initialize the metric with RNPTC as the default method
    metric = MaskedProbabilityClassifier(method="rnptc", num_masks=10)
    
    technical_text = """
    Simple bone cysts are typically benign. However, differential diagnosis must include potentially malignant lesions. Primary bone malignancies may present with cystic features. Key distinctions via imaging characteristics, location, patient age, and growth pattern. Biopsy indicated for atypical presentations, aggressive features, or uncertain diagnosis. Common malignant mimics: telangiectatic osteosarcoma, giant cell tumor, aneurysmal bone cyst.    
    """
    
    plain_text = """
    While most bone cysts are benign (non-cancerous), some cyst-like lesions in bones can be cancerous. Your doctor will use X-rays, MRI, or other imaging tests to determine if a bone cyst needs further investigation. If there are any concerning features, they may recommend a biopsy to confirm it's benign.
    """
    
    print("===== Using Default Method (RNPTC) =====")
    
    # These will use RNPTC method by default
    technical_score = metric.predict_single(technical_text)
    plain_score = metric.predict_single(plain_text)
    
    print(f"Technical text score: {technical_score:.4f}")
    print(f"Plain text score: {plain_score:.4f}")
    print(f"Score difference: {technical_score - plain_score:.4f}")
