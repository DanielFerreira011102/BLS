"""
AlignScore: A cleaner implementation of the alignment scoring metric that's compatible with newer versions
of protobuf and PyTorch while producing the same results as the original implementation.
"""

import logging
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Union, Optional
from pathlib import Path
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AdamW, 
    get_linear_schedule_with_warmup,
    BertForPreTraining, 
    BertModel, 
    RobertaModel, 
    AlbertModel, 
    AlbertForMaskedLM, 
    RobertaForMaskedLM
)
import pytorch_lightning as pl
from dataclasses import dataclass
from sklearn.metrics import f1_score
import math


@dataclass
class ModelOutput:
    """Container for model outputs with clear typing."""
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[List] = None
    loss_nums: Optional[List] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.gelu(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits


class BERTAlignModel(pl.LightningModule):
    """
    BERT-based model for alignment scoring with multiple output heads.
    """
    
    def __init__(
        self, 
        model: str = 'bert-base-uncased', 
        using_pretrained: bool = True,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        adam_epsilon: float = 1e-8,
        warmup_steps_portion: float = 0.1,
        **kwargs
    ) -> None:
        """
        Initialize the BERTAlignModel.
        
        Args:
            model: Base model name
            using_pretrained: Whether to use pretrained weights
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for optimization
            adam_epsilon: Epsilon for Adam optimizer
            warmup_steps_portion: Portion of steps to use for warmup
        """
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model
        
        # Initialize model components based on model type
        self._initialize_base_model(model, using_pretrained)
        
        # Classification heads
        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)
        self.dropout = nn.Dropout(p=0.1)
        
        # Training settings
        self.need_mlm = True
        self.is_finetune = False
        self.mlm_loss_factor = 0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def _initialize_base_model(self, model: str, using_pretrained: bool) -> None:
        """
        Initialize the appropriate model architecture based on model name.
        
        Args:
            model: Model name/path
            using_pretrained: Whether to use pretrained weights
        """
        if 'muppet' in model:
            assert using_pretrained, "Only support pretrained muppet!"
            self.base_model = RobertaModel.from_pretrained(model)
            self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head
            
        elif 'roberta' in model:
            if using_pretrained:
                self.base_model = RobertaModel.from_pretrained(model)
                self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
            else:
                self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
                self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head
            
        elif 'albert' in model:
            if using_pretrained:
                self.base_model = AlbertModel.from_pretrained(model)
                self.mlm_head = AlbertForMaskedLM.from_pretrained(model).predictions
            else:
                self.base_model = AlbertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = AlbertForMaskedLM(AutoConfig.from_pretrained(model)).predictions
            
        elif 'bert' in model:
            if using_pretrained:
                self.base_model = BertModel.from_pretrained(model)
                self.mlm_head = BertForPreTraining.from_pretrained(model).cls.predictions
            else:
                self.base_model = BertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = BertForPreTraining(AutoConfig.from_pretrained(model)).cls.predictions

        elif 'electra' in model:
            self.generator = BertModel(AutoConfig.from_pretrained('prajjwal1/bert-small'))
            self.generator_mlm = BertForPreTraining(AutoConfig.from_pretrained('prajjwal1/bert-small')).cls.predictions
            self.base_model = BertModel(AutoConfig.from_pretrained('bert-base-uncased'))
            self.discriminator_predictor = ElectraDiscriminatorPredictions(self.base_model.config)
        else:
            raise ValueError(f"Unsupported model type: {model}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            batch: Input batch with tensors
            
        Returns:
            ModelOutput with various head outputs
        """
        if 'electra' in self.model_name:
            return self._electra_forward(batch)
        
        # Handle token type IDs for models that support them
        token_type_ids = batch.get('token_type_ids', None)
        
        # Get outputs from base model
        base_model_output = self.base_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=token_type_ids
        )
        
        # Apply various prediction heads
        prediction_scores = self.mlm_head(base_model_output.last_hidden_state)
        seq_relationship_score = self.bin_layer(self.dropout(base_model_output.pooler_output))
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        # Calculate losses if labels are provided
        total_loss = None
        masked_lm_loss = next_sentence_loss = tri_label_loss = reg_label_loss = None
        masked_lm_loss_num = next_sentence_loss_num = tri_label_loss_num = reg_label_loss_num = None
        
        if 'mlm_label' in batch:
            ce_loss_fct = nn.CrossEntropyLoss(reduction='sum')
            masked_lm_loss = ce_loss_fct(
                prediction_scores.view(-1, self.base_model.config.vocab_size), 
                batch['mlm_label'].view(-1)
            )
            next_sentence_loss = ce_loss_fct(
                seq_relationship_score.view(-1, 2), 
                batch['align_label'].view(-1)
            ) / math.log(2)
            tri_label_loss = ce_loss_fct(
                tri_label_score.view(-1, 3), 
                batch['tri_label'].view(-1)
            ) / math.log(3)
            reg_label_loss = self._mse_loss(
                reg_label_score.view(-1), 
                batch['reg_label'].view(-1), 
                reduction='sum'
            )

            masked_lm_loss_num = torch.sum(batch['mlm_label'].view(-1) != -100)
            next_sentence_loss_num = torch.sum(batch['align_label'].view(-1) != -100)
            tri_label_loss_num = torch.sum(batch['tri_label'].view(-1) != -100)
            reg_label_loss_num = torch.sum(batch['reg_label'].view(-1) != -100.0)

        return ModelOutput(
            loss=total_loss,
            all_loss=[masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss] 
                if 'mlm_label' in batch else None,
            loss_nums=[masked_lm_loss_num, next_sentence_loss_num, tri_label_loss_num, reg_label_loss_num] 
                if 'mlm_label' in batch else None,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )

    def _electra_forward(self, batch: Dict[str, torch.Tensor]) -> ModelOutput:
        """
        Forward pass for Electra model variant.
        
        Args:
            batch: Input batch with tensors
            
        Returns:
            ModelOutput with various head outputs
        """
        # Handle token type IDs
        token_type_ids = batch.get('token_type_ids', None)
        
        # Generator phase (if training)
        masked_lm_loss = hallucinated_tokens = None
        if 'mlm_label' in batch:
            ce_loss_fct = nn.CrossEntropyLoss()
            generator_output = self.generator(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=token_type_ids
            )
            
            generator_output = self.generator_mlm(generator_output.last_hidden_state)
            masked_lm_loss = ce_loss_fct(
                generator_output.view(-1, self.generator.config.vocab_size), 
                batch['mlm_label'].view(-1)
            )

            # Create inputs for discriminator
            hallucinated_tokens = batch['input_ids'].clone()
            mask_indices = (batch['mlm_label'] != -100)
            hallucinated_tokens[mask_indices] = torch.argmax(
                generator_output, dim=-1
            )[mask_indices]
            
            # Create replaced token labels for discriminator
            replaced_token_label = (batch['input_ids'] == hallucinated_tokens).long()
            replaced_token_label[mask_indices] = (
                batch['mlm_label'] == hallucinated_tokens
            )[mask_indices].long()
            replaced_token_label[batch['input_ids'] == 0] = -100  # Ignore paddings
        
        # Discriminator phase
        base_model_output = self.base_model(
            input_ids=hallucinated_tokens if hallucinated_tokens is not None else batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=token_type_ids
        )
        
        hallu_detect_score = self.discriminator_predictor(base_model_output.last_hidden_state)
        seq_relationship_score = self.bin_layer(self.dropout(base_model_output.pooler_output))
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        # Calculate losses if training
        total_loss = None
        hallu_detect_loss = next_sentence_loss = tri_label_loss = reg_label_loss = None
        
        if 'mlm_label' in batch:
            ce_loss_fct = nn.CrossEntropyLoss()
            hallu_detect_loss = ce_loss_fct(
                hallu_detect_score.view(-1, 2),
                replaced_token_label.view(-1)
            )
            next_sentence_loss = ce_loss_fct(
                seq_relationship_score.view(-1, 2), 
                batch['align_label'].view(-1)
            )
            tri_label_loss = ce_loss_fct(
                tri_label_score.view(-1, 3), 
                batch['tri_label'].view(-1)
            )
            reg_label_loss = self._mse_loss(
                reg_label_score.view(-1), 
                batch['reg_label'].view(-1)
            )

            # Combine losses with appropriate handling of NaNs
            losses = []
            if not torch.isnan(hallu_detect_loss).item():
                losses.append(10.0 * hallu_detect_loss)
            if not torch.isnan(masked_lm_loss).item() and self.need_mlm:
                losses.append(0.2 * masked_lm_loss)
            if not torch.isnan(next_sentence_loss).item():
                losses.append(next_sentence_loss)
            if not torch.isnan(tri_label_loss).item():
                losses.append(tri_label_loss)
            if not torch.isnan(reg_label_loss).item():
                losses.append(reg_label_loss)
                
            total_loss = sum(losses) if losses else torch.tensor(0.0, device=batch['input_ids'].device)

        return ModelOutput(
            loss=total_loss,
            all_loss=[masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss, hallu_detect_loss] 
                if 'mlm_label' in batch else None,
            prediction_logits=hallu_detect_score,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )

    def training_step(self, train_batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Process a single training step."""
        output = self(train_batch)
        return {'losses': output.all_loss, 'loss_nums': output.loss_nums}

    def training_step_end(self, step_output: Dict[str, List]) -> torch.Tensor:
        """Process the end of a training step."""
        losses = step_output['losses']
        loss_nums = step_output['loss_nums']
        assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'

        # Calculate weighted losses
        loss_mlm_num = torch.sum(loss_nums[0])
        loss_bin_num = torch.sum(loss_nums[1])
        loss_tri_num = torch.sum(loss_nums[2])
        loss_reg_num = torch.sum(loss_nums[3])

        loss_mlm = torch.sum(losses[0]) / loss_mlm_num if loss_mlm_num > 0 else 0.
        loss_bin = torch.sum(losses[1]) / loss_bin_num if loss_bin_num > 0 else 0.
        loss_tri = torch.sum(losses[2]) / loss_tri_num if loss_tri_num > 0 else 0.
        loss_reg = torch.sum(losses[3]) / loss_reg_num if loss_reg_num > 0 else 0.

        total_loss = self.mlm_loss_factor * loss_mlm + loss_bin + loss_tri + loss_reg

        # Log losses
        self.log('train_loss', total_loss)
        self.log('mlm_loss', loss_mlm)
        self.log('bin_label_loss', loss_bin)
        self.log('tri_label_loss', loss_tri)
        self.log('reg_label_loss', loss_reg)

        return total_loss
    
    def validation_step(self, val_batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """Process a single validation step."""
        if not self.is_finetune:
            with torch.no_grad():
                output = self(val_batch)
            return {'losses': output.all_loss, 'loss_nums': output.loss_nums}

        # Fine-tuning mode - compute predictions
        with torch.no_grad():
            output = self(val_batch)
            probs = self.softmax(output.seq_relationship_logits)[:, 1].tolist()
            pred = [int(align_prob > 0.5) for align_prob in probs]
            labels = val_batch['align_label'].tolist()

        return {"pred": pred, 'labels': labels}

    def validation_step_end(self, step_output: Dict[str, Any]) -> torch.Tensor:
        """Process the end of a validation step."""
        # Skip if in finetune mode and no losses
        if 'losses' not in step_output:
            return torch.tensor(0.0)
            
        losses = step_output['losses']
        loss_nums = step_output['loss_nums']
        
        # Calculate weighted losses
        loss_mlm_num = torch.sum(loss_nums[0])
        loss_bin_num = torch.sum(loss_nums[1])
        loss_tri_num = torch.sum(loss_nums[2])
        loss_reg_num = torch.sum(loss_nums[3])

        loss_mlm = torch.sum(losses[0]) / loss_mlm_num if loss_mlm_num > 0 else 0.
        loss_bin = torch.sum(losses[1]) / loss_bin_num if loss_bin_num > 0 else 0.
        loss_tri = torch.sum(losses[2]) / loss_tri_num if loss_tri_num > 0 else 0.
        loss_reg = torch.sum(losses[3]) / loss_reg_num if loss_reg_num > 0 else 0.

        total_loss = self.mlm_loss_factor * loss_mlm + loss_bin + loss_tri + loss_reg

        # Log losses
        self.log('val_mlm_loss', loss_mlm)
        self.log('val_bin_label_loss', loss_bin)
        self.log('val_tri_label_loss', loss_tri)
        self.log('val_reg_label_loss', loss_reg)

        return total_loss

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        """Process the end of a validation epoch."""
        if not self.is_finetune:
            # Calculate average loss
            losses = [o for o in outputs if isinstance(o, torch.Tensor)]
            if losses:
                total_loss = torch.stack(losses).mean()
                self.log("val_loss", total_loss, prog_bar=True)
            return
        
        # Calculate F1 score for finetuning
        all_predictions = []
        all_labels = []
        for each_output in outputs:
            if 'pred' in each_output:
                all_predictions.extend(each_output['pred'])
                all_labels.extend(each_output['labels'])
        
        if all_predictions:
            f1 = f1_score(all_labels, all_predictions)
            self.log("f1", f1, prog_bar=True)

    def configure_optimizers(self) -> Tuple[List, List]:
        """Configure optimizers and learning rate schedulers."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # PyTorch 2.x compatible optimizer
        optimizer = AdamW(
            optimizer_grouped_parameters, 
            lr=self.hparams.learning_rate, 
            eps=self.hparams.adam_epsilon
        )

        # Get the number of training steps (PyTorch Lightning 2.x compatible)
        try:
            # For newer Lightning versions
            stepping_batches = self.trainer.estimated_stepping_batches
        except AttributeError:
            # Fallback for older versions or compatibility
            stepping_batches = getattr(self.trainer, "max_steps", 1000)
        
        # Create learning rate scheduler
        warmup_steps = max(1, int(self.hparams.warmup_steps_portion * stepping_batches))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=stepping_batches,
        )
        scheduler_config = {
            "scheduler": scheduler, 
            "interval": "step", 
            "frequency": 1
        }
        
        return [optimizer], [scheduler_config]

    def _mse_loss(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor, 
        ignored_index: float = -100.0, 
        reduction: str = 'mean'
    ) -> torch.Tensor:
        """
        Custom MSE loss that handles ignored indices.
        
        Args:
            input: Predicted values
            target: Target values
            ignored_index: Index value to ignore
            reduction: Reduction method ('mean' or 'sum')
            
        Returns:
            Loss value
        """
        mask = (target == ignored_index)
        if mask.all():
            return torch.tensor(0.0, device=input.device)
            
        squared_diff = (input[~mask] - target[~mask]) ** 2
        
        if reduction == "mean":
            return squared_diff.mean()
        elif reduction == "sum":
            return squared_diff.sum()
        else:
            return squared_diff


class Inferencer:
    """
    Handles inference for AlignScore, computing alignment between premise and hypothesis.
    """
    
    def __init__(
        self, 
        ckpt_path: Optional[str] = None, 
        model: str = "bert-base-uncased", 
        batch_size: int = 32, 
        device: Union[str, int] = "cuda" if torch.cuda.is_available() else "cpu", 
        verbose: bool = True
    ) -> None:
        """
        Initialize the inferencer.
        
        Args:
            ckpt_path: Path to the model checkpoint
            model: Base model name
            batch_size: Batch size for inference
            device: Device to use for inference
            verbose: Whether to display progress bars
        """
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        
        # Load model
        if ckpt_path is not None:
            self.model = BERTAlignModel(model=model).load_from_checkpoint(
                checkpoint_path=ckpt_path, strict=False
            ).to(self.device)
        else:
            logging.warning('Loading UNTRAINED model!')
            self.model = BERTAlignModel(model=model).to(self.device)
        
        self.model.eval()
        
        # Load tokenizer and other components
        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.spacy_model = spacy.load('en_core_web_sm')
        
        # Set up loss and activation functions
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=-1)
        
        # Default settings
        self.smart_type = 'smart-n'
        self.smart_n_metric = 'f1'
        self.disable_progress_bar_in_inference = False
        self.nlg_eval_mode = None
    
    def nlg_eval(self, premise: List[str], hypo: List[str]) -> Tuple:
        """
        Evaluate alignment between premise and hypothesis.
        
        Args:
            premise: List of premise (context) strings
            hypo: List of hypothesis (claim) strings
            
        Returns:
            Tuple of scores depending on the evaluation mode
        """
        if not self.nlg_eval_mode:
            raise ValueError("Select NLG Eval mode!")
            
        if self.nlg_eval_mode in ['bin', 'nli', 'reg']:
            return self.inference(premise, hypo)
        
        elif self.nlg_eval_mode in ['bin_sp', 'nli_sp', 'reg_sp']:
            return self.inference_example_batch(premise, hypo)
        
        else:
            raise ValueError(f"Unrecognized NLG Eval mode: {self.nlg_eval_mode}")
    
    def inference_example_batch(self, premise: List[str], hypo: List[str]) -> Tuple:
        """
        Run inference on batches of examples using SummaC style aggregation.
        
        Args:
            premise: List of premise strings
            hypo: List of hypothesis strings
            
        Returns:
            Tuple of (None, scores_tensor, None)
        """
        self.disable_progress_bar_in_inference = True
        if len(premise) != len(hypo):
            raise ValueError("Premise must have the same length as Hypothesis!")

        out_score = []
        for one_pre, one_hypo in tqdm(
            zip(premise, hypo), 
            desc="Evaluating", 
            total=len(premise), 
            disable=(not self.verbose)
        ):
            out_score.append(self.inference_per_example(one_pre, one_hypo))
        
        return None, torch.tensor(out_score), None
    
    def inference_per_example(self, premise: str, hypo: str) -> float:
        """
        Calculate alignment score for a single premise-hypothesis pair.
        
        Args:
            premise: Premise (context) string
            hypo: Hypothesis (claim) string
            
        Returns:
            Alignment score
        """
        # Split input texts into sentences
        premise_sents = sent_tokenize(premise.strip()) or ['']
        hypo_sents = sent_tokenize(hypo.strip())
        
        # Determine chunk size based on text length
        n_words = len(premise.strip().split())
        n_chunk = max(1, n_words // 350 + 1)
        n_chunk = max(1, len(premise_sents) // n_chunk)
        
        # Chunk premise sentences to avoid overly long sequences
        premise_sents = self._chunk_sentences(premise_sents, n_chunk)
        
        # Create premise-hypothesis sentence pairs
        premise_sent_mat = []
        hypo_sents_mat = []
        for p_sent in premise_sents:
            for h_sent in hypo_sents:
                premise_sent_mat.append(p_sent)
                hypo_sents_mat.append(h_sent)
        
        # Handle empty case
        if not premise_sent_mat or not hypo_sents_mat:
            return 0.0
            
        # Get scores based on evaluation mode
        if self.nlg_eval_mode:
            output = self.inference(premise_sent_mat, hypo_sents_mat)
            
            if self.nlg_eval_mode == 'nli_sp':
                output_score = output[2][:, 0]  # NLI head
            elif self.nlg_eval_mode == 'bin_sp':
                output_score = output[1]  # Binary head
            elif self.nlg_eval_mode == 'reg_sp':
                output_score = output[0]  # Regression head
            else:
                # Default to NLI head
                output_score = output[2][:, 0]
            
            # Reshape and aggregate
            output_score = output_score.view(len(premise_sents), len(hypo_sents))
            return output_score.max(dim=0).values.mean().item()
        
        # Default behavior (same as nli_sp)
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:, 0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))
        return output_score.max(dim=0).values.mean().item()

    def inference(self, premise: Union[str, List[str]], hypo: Union[str, List[str]]) -> Tuple:
        """
        Run inference on premise-hypothesis pairs.
        
        Args:
            premise: Single premise string or list of premise strings
            hypo: Single hypothesis string or list of hypothesis strings
            
        Returns:
            Tuple of (regression_scores, binary_scores, nli_scores)
        """
        # Convert single strings to lists
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        # Tokenize and batch
        batches = self._batch_tokenize(premise, hypo)
        
        # Process each batch
        output_score_reg = []
        output_score_bin = []
        output_score_tri = []

        for mini_batch in tqdm(
            batches, 
            desc="Evaluating", 
            disable=not self.verbose or self.disable_progress_bar_in_inference
        ):
            mini_batch = mini_batch.to(self.device)
            
            with torch.no_grad():
                model_output = self.model(mini_batch)
                
                # Extract and process output logits
                model_output_reg = model_output.reg_label_logits.cpu()
                model_output_bin = model_output.seq_relationship_logits
                model_output_tri = model_output.tri_label_logits
                
                # Apply softmax for classification outputs
                model_output_bin = self.softmax(model_output_bin).cpu()
                model_output_tri = self.softmax(model_output_tri).cpu()
                
            # Collect scores
            output_score_reg.append(model_output_reg[:, 0])
            output_score_bin.append(model_output_bin[:, 1])
            output_score_tri.append(model_output_tri[:, :])
        
        # Concatenate results
        output_score_reg = torch.cat(output_score_reg)
        output_score_bin = torch.cat(output_score_bin)
        output_score_tri = torch.cat(output_score_tri)
        
        # Return specific scores based on evaluation mode
        if self.nlg_eval_mode:
            if self.nlg_eval_mode == 'nli':
                return None, output_score_tri[:, 0], None
            elif self.nlg_eval_mode == 'bin':
                return None, output_score_bin, None
            elif self.nlg_eval_mode == 'reg':
                return None, output_score_reg, None
        
        # Default: return all scores
        return output_score_reg, output_score_bin, output_score_tri
    
    def _batch_tokenize(self, premise: List[str], hypo: List[str]) -> List[Dict[str, torch.Tensor]]:
        """
        Tokenize and batch premise-hypothesis pairs.
        
        Args:
            premise: List of premise strings
            hypo: List of hypothesis strings
            
        Returns:
            List of batched inputs for the model
        """
        if not isinstance(premise, list) or not isinstance(hypo, list):
            raise TypeError("Inputs must be lists")
            
        if len(premise) != len(hypo):
            raise ValueError("Premise and hypothesis should have the same length")

        batches = []
        premise_chunks = self._chunk_list(premise, self.batch_size)
        hypo_chunks = self._chunk_list(hypo, self.batch_size)
        
        for mini_batch_pre, mini_batch_hypo in zip(premise_chunks, hypo_chunks):
            try:
                # Try with truncation of only the first sequence
                mini_batch = self.tokenizer(
                    mini_batch_pre, 
                    mini_batch_hypo, 
                    truncation='only_first', 
                    padding='max_length', 
                    max_length=self.tokenizer.model_max_length, 
                    return_tensors='pt'
                )
            except Exception as e:
                # Fall back to truncating both sequences if needed
                logging.warning(f'Text_b too long: {e}')
                mini_batch = self.tokenizer(
                    mini_batch_pre, 
                    mini_batch_hypo, 
                    truncation=True, 
                    padding='max_length', 
                    max_length=self.tokenizer.model_max_length, 
                    return_tensors='pt'
                )
            batches.append(mini_batch)

        return batches
    
    def _chunk_list(self, lst: List[Any], chunk_size: int) -> List[List[Any]]:
        """
        Split a list into chunks of specified size.
        
        Args:
            lst: List to chunk
            chunk_size: Maximum chunk size
            
        Returns:
            List of chunks
        """
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
        
    def _chunk_sentences(self, sentences: List[str], chunk_size: int) -> List[str]:
        """
        Join sentences into chunks.
        
        Args:
            sentences: List of sentence strings
            chunk_size: Number of sentences per chunk
            
        Returns:
            List of chunked sentence strings
        """
        return [
            ' '.join(sentences[i:i + chunk_size]) 
            for i in range(0, len(sentences), chunk_size)
        ]
        
    # SMART methods (not used in the main alignment flow but preserved for compatibility)
    def smart_doc(self, premise: List[str], hypo: List[str]) -> Tuple:
        """
        SMART style aggregation for document-level alignment.
        
        Args:
            premise: List of premise texts
            hypo: List of hypothesis texts
            
        Returns:
            Tuple of (None, scores_tensor, None)
        """
        self.disable_progress_bar_in_inference = True
        if len(premise) != len(hypo):
            raise ValueError("Premise must have the same length as Hypothesis!")
            
        if self.smart_type not in ['smart-n', 'smart-l']:
            raise ValueError(f"Unsupported SMART type: {self.smart_type}")

        out_score = []
        for one_pre, one_hypo in tqdm(
            zip(premise, hypo), 
            desc="Evaluating SMART", 
            total=len(premise), 
            disable=not self.verbose
        ):
            if self.smart_type == 'smart-l':
                out_score.append(self.smart_l(one_pre, one_hypo)[1])
            else:
                out_score.append(self.smart_n(one_pre, one_hypo)[1])
        
        return None, torch.tensor(out_score), None

    def smart_l(self, premise: str, hypo: str) -> Tuple:
        """
        SMART-L algorithm for alignment scoring.
        
        Args:
            premise: Premise text
            hypo: Hypothesis text
            
        Returns:
            Tuple of (None, score, None)
        """
        # Extract sentences using spaCy
        premise_sents = [each.text for each in self.spacy_model(premise).sents]
        hypo_sents = [each.text for each in self.spacy_model(hypo).sents]
        
        if not premise_sents or not hypo_sents:
            return None, 0.0, None

        # Create premise-hypothesis sentence pairs
        premise_sent_mat = []
        hypo_sents_mat = []
        for p_sent in premise_sents:
            for h_sent in hypo_sents:
                premise_sent_mat.append(p_sent)
                hypo_sents_mat.append(h_sent)
        
        # Get raw scores
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:, 0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))

        # Compute LCS-based alignment score
        lcs = [[0] * (len(hypo_sents) + 1) for _ in range(len(premise_sents) + 1)]
        for i in range(1, len(premise_sents) + 1):
            for j in range(1, len(hypo_sents) + 1):
                m = output_score[i-1, j-1]
                lcs[i][j] = max(
                    lcs[i-1][j-1] + m,
                    lcs[i-1][j] + m,
                    lcs[i][j-1]
                )

        # Normalize by premise length
        score = lcs[-1][-1] / len(premise_sents) if premise_sents else 0.0
        return None, score, None
    
    def smart_n(self, premise: str, hypo: str) -> Tuple:
        """
        SMART-N algorithm for alignment scoring.
        
        Args:
            premise: Premise text
            hypo: Hypothesis text
            
        Returns:
            Tuple of (None, score, None)
        """
        # Fixed n-gram size
        n_gram = 1

        # Extract sentences using spaCy
        premise_sents = [each.text for each in self.spacy_model(premise).sents]
        hypo_sents = [each.text for each in self.spacy_model(hypo).sents]
        
        if not premise_sents or not hypo_sents:
            return None, 0.0, None

        # Create premise-hypothesis sentence pairs
        premise_sent_mat = []
        hypo_sents_mat = []
        for p_sent in premise_sents:
            for h_sent in hypo_sents:
                premise_sent_mat.append(p_sent)
                hypo_sents_mat.append(h_sent)
        
        # Get raw scores for precision
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:, 0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))
        
        # Calculate precision
        prec_sum = 0
        valid_prec_items = 0
        for j in range(len(hypo_sents) - n_gram + 1):
            max_prec = 0
            for i in range(len(premise_sents) - n_gram + 1):
                ngram_score = sum(output_score[i+n, j+n] / n_gram for n in range(n_gram))
                max_prec = max(max_prec, ngram_score)
            prec_sum += max_prec
            valid_prec_items += 1
        
        prec = prec_sum / valid_prec_items if valid_prec_items > 0 else 0.0

        # Swap premise and hypothesis for recall
        premise_sents, hypo_sents = hypo_sents, premise_sents

        # Create new premise-hypothesis sentence pairs for recall
        premise_sent_mat = []
        hypo_sents_mat = []
        for p_sent in premise_sents:
            for h_sent in hypo_sents:
                premise_sent_mat.append(p_sent)
                hypo_sents_mat.append(h_sent)
        
        # Get raw scores for recall
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:, 0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))

        # Calculate recall
        recall_sum = 0
        valid_recall_items = 0
        for j in range(len(hypo_sents) - n_gram + 1):
            max_recall = 0
            for i in range(len(premise_sents) - n_gram + 1):
                ngram_score = sum(output_score[i+n, j+n] / n_gram for n in range(n_gram))
                max_recall = max(max_recall, ngram_score)
            recall_sum += max_recall
            valid_recall_items += 1
        
        recall = recall_sum / valid_recall_items if valid_recall_items > 0 else 0.0

        # Calculate F1
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

        # Return the requested metric
        if self.smart_n_metric == 'f1':
            return None, f1, None
        elif self.smart_n_metric == 'precision':
            return None, prec, None
        elif self.smart_n_metric == 'recall':
            return None, recall, None
        else:
            raise ValueError(f"Unsupported SMART-N metric: {self.smart_n_metric}")


class AlignScore:
    """
    AlignScore measures the semantic alignment between contexts and claims.
    
    This class provides a simple interface to the underlying BERT-based alignment model, 
    producing scores that indicate how well claims are supported by their contexts.
    """
    
    def __init__(
        self, 
        model: str = "bert-base-uncased", 
        batch_size: int = 32, 
        device: Union[str, int] = "cuda" if torch.cuda.is_available() else "cpu", 
        ckpt_path: Optional[Union[str, Path]] = None, 
        evaluation_mode: str = "nli_sp", 
        verbose: bool = True
    ) -> None:
        """
        Initialize the AlignScore model.
        
        Args:
            model: Model name or path (e.g., "bert-base-uncased")
            batch_size: Batch size for inference
            device: Device to use ("cuda", "cpu", or device index)
            ckpt_path: Path to model checkpoint file
            evaluation_mode: Scoring method - one of:
                - "nli_sp": NLI-based scoring with sentence-level processing (default)
                - "bin_sp": Binary classification with sentence-level processing
                - "reg_sp": Regression with sentence-level processing
                - "nli": NLI-based scoring without sentence-level processing
                - "bin": Binary classification without sentence-level processing
                - "reg": Regression without sentence-level processing
            verbose: Whether to show progress bars
        """
        # Convert path to string if needed
        if isinstance(ckpt_path, Path):
            ckpt_path = str(ckpt_path)
            
        # Convert device to string if it's an integer
        if isinstance(device, int):
            device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        
        # Initialize the Inferencer
        self.model = Inferencer(
            ckpt_path=ckpt_path, 
            model=model,
            batch_size=batch_size, 
            device=device,
            verbose=verbose
        )
        
        # Set evaluation mode
        valid_modes = ["nli_sp", "bin_sp", "reg_sp", "nli", "bin", "reg"]
        if evaluation_mode not in valid_modes:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}. Must be one of {valid_modes}")
            
        self.model.nlg_eval_mode = evaluation_mode
    
    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        """
        Score the alignment between contexts and claims.
        
        Higher scores indicate better alignment (claims are well-supported by contexts).
        
        Args:
            contexts: List of context texts
            claims: List of claim texts to evaluate against the contexts
            
        Returns:
            List of alignment scores (typically between 0 and 1)
        """
        if len(contexts) != len(claims):
            raise ValueError("Number of contexts and claims must be equal")
            
        # Handle empty inputs
        if not contexts or not claims:
            return []
            
        # Process and return scores
        return self.model.nlg_eval(contexts, claims)[1].tolist()