import json
import os
import argparse
from pathlib import Path
from typing import Dict, Any, Union, List

import torch
from transformers import AutoTokenizer

from rq1.llmify.llmify import (
    ReadabilityModel,
    ImprovedReadabilityModel,
    BaseReadabilityModel,
    DIMENSIONS,
)

from utils.helpers import setup_logging, load_json, save_json

logger = setup_logging()


class LLMReadabilityClassifier:
    """Light-weight wrapper around a trained readability model."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = self._load_config()

        # Set up tokenizer
        model_name = self.config.get("model_name", "kamalkraj/BioSimCSE-BioLinkBERT-BASE")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)

        # Load the best checkpoint
        self.model = self._build_model().to(self.device).eval()
        self._load_weights(self.model)

        logger.info("Model loaded from %s on %s", self.model_path, self.device)


    def _load_config(self) -> Dict[str, Any]:
        cfg_file = self.model_path / "config.json"

        if cfg_file.exists():
            return load_json(cfg_file)

        # Fallback: extract from the first checkpoint
        ckpt_files = list(self.model_path.glob("*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints in {self.model_path}")

        checkpoint = torch.load(ckpt_files[0], map_location="cpu")
        config = checkpoint["config"]

        # Cache for next time
        save_json(config, cfg_file)
        return config


    def _best_checkpoint(self) -> Path:
        best = self.model_path / "best_model.pt"
        if best.exists():
            return best

        final = self.model_path / "final_model.pt"
        if final.exists():
            return final

        ckpts = list(self.model_path.glob("*.pt"))
        if not ckpts:
            raise FileNotFoundError("No .pt files found")
        return ckpts[0]


    def _build_model(self) -> BaseReadabilityModel:
        model_name = self.config["model_name"]
        model_type = self.config.get("model_type", "standard")
        heads = self.config.get("attention_heads", 2)
        dropout = self.config.get("dropout_rate", 0.2)

        if model_type == "improved":
            return ImprovedReadabilityModel(
                model_name,
                tokenizer=self.tokenizer,
                attention_heads=heads,
                dropout_rate=dropout,
            )

        return ReadabilityModel(
            model_name, tokenizer=self.tokenizer, dropout_rate=dropout
        )


    def _load_weights(self, model: BaseReadabilityModel) -> None:
        checkpoint = self._best_checkpoint()
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        logger.info("Weights loaded from %s", checkpoint)


    def predict_single(self, text: str, max_length: int = 512) -> Dict[str, Any]:
        return self.model.predict_single(text, max_length=max_length)


    def predict_batch(self, texts: List[str], batch_size: int = 8, max_length: int = 512) -> List[Dict[str, Any]]:
        return self.model.predict_batch(texts, batch_size=batch_size, max_length=max_length)


    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 8,
        max_length: int = 512,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        if isinstance(texts, str):
            return self.predict_single(texts, max_length=max_length)
        return self.predict_batch(texts, batch_size=batch_size, max_length=max_length)


def load_data(args) -> str:
    """Read text from command line argument or file."""
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    raise ValueError("Either --text or --file must be provided")


def print_result(result: Dict[str, Any], source: str = "") -> None:
    """Format and print the readability scores."""
    if source:
        print(f"\nReadability Scores for: {source}")
    else:
        print("\nReadability Scores:")

    print(f"Overall Score : {result['overall_score']:.2f}/5.00")
    print("\nDimension Scores:")
    for dim, score in result["dimension_scores"].items():
        print(f"  {dim.replace('_', ' ').title():<25} {score:.2f}/5.00")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text readability.")
    parser.add_argument("--model-path", required=True, help="Directory with saved model")
    parser.add_argument("--text", help="Text to score")
    parser.add_argument("--file", help="Path to text file to score")
    args = parser.parse_args()
    
    text = load_data(args)
    classifier = LLMReadabilityClassifier(args.model_path)
    result = classifier.predict(text)

    print_result(result, source=args.file or None)


if __name__ == "__main__":
    main()