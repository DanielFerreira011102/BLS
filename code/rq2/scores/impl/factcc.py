import torch
from transformers import BertForSequenceClassification, BertTokenizer

class FactCC:
    """Implementation of FactCC for factual consistency checking."""
    
    def __init__(self, model_name="manueldeprada/FactCC", device=None, batch_size=8, max_length=512):
        """Initialize FactCC model and tokenizer."""
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get correct label index
        self.correct_idx = None
        for label, idx in self.model.config.label2id.items():
            if label == "CORRECT":
                self.correct_idx = idx
                break
        
        # If CORRECT label not found, default to index 1
        if self.correct_idx is None:
            self.correct_idx = 1
    
    def score(self, contexts, claims):
        """
        Score the factual consistency of claims against contexts.
        
        Args:
            contexts: List of reference texts
            claims: List of texts to check for factuality
            
        Returns:
            List of factuality scores (1.0 = fully factual, 0.0 = not factual)
        """
        all_scores = []
        
        for i in range(0, len(claims), self.batch_size):
            batch_contexts = contexts[i:i+self.batch_size]
            batch_claims = claims[i:i+self.batch_size]
            
            with torch.no_grad():
                # Tokenize batch input with appropriate settings
                # Using 'longest_first' strategy to handle texts of any length
                inputs = self.tokenizer(
                    batch_contexts, 
                    batch_claims, 
                    max_length=self.max_length, 
                    truncation=True,
                    padding=True,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get probabilities using softmax
                probs = torch.nn.functional.softmax(logits, dim=1)
                
                # Get score (probability of CORRECT) for each pair
                batch_scores = [prob[self.correct_idx].item() for prob in probs]
                all_scores.extend(batch_scores)
        
        return all_scores