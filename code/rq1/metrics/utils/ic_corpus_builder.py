import re
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Union

import spacy
import json
import datasets

from utils.helpers import setup_logging

logger = setup_logging()


class CorpusBuilder:
    """Builds a corpus from multiple medical datasets."""
    
    def __init__(self, min_words: int = 20):
        self.min_words = min_words

    def _filter_text(self, text: str) -> Optional[str]:
        """Centralized text filtering logic."""
        if not text or not isinstance(text, str):
            return None

        # Clean text formatting
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = text.encode("ascii", errors="replace").decode()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()

        # Filter by minimum word count
        if len(text.split()) < self.min_words:
            return None
            
        return text

    def _extract_field_text(self, item: dict, field: str) -> Optional[str]:
        """Extract and filter text from a single field."""
        if field not in item:
            return None
            
        field_text = item.get(field, '')
        if not field_text or not isinstance(field_text, str):
            return None
            
        return self._filter_text(field_text)

    def _process_dataset_split(self, dataset_split, text_fields: List[str]) -> List[str]:
        """Process a single dataset split."""
        texts = []
        
        for item in dataset_split:
            for field in text_fields:
                filtered_text = self._extract_field_text(item, field)
                if filtered_text:
                    texts.append(filtered_text)
        
        return texts

    def _process_dataset(self, dataset: datasets.Dataset, name: str, text_fields: Union[str, List[str]]) -> List[str]:
        """Process a single dataset with support for multiple text fields."""
        # Normalize text_fields to list
        if isinstance(text_fields, str):
            text_fields = [text_fields]
        
        texts = []
        total_items = sum(len(split) for split in dataset.values())
        
        for split_name in dataset:
            logger.info(f"Processing split '{split_name}' with {len(dataset[split_name])} items")
            split_texts = self._process_dataset_split(dataset[split_name], text_fields)
            texts.extend(split_texts)
        
        logger.info(f"Kept {len(texts)} texts from fields: {text_fields}")
        return texts

    def _load_dataset_safely(self, dataset_path: str, name: str, text_fields: Union[str, List[str]]) -> List[str]:
        """Load a dataset safely with error handling."""
        dataset = datasets.load_dataset(dataset_path)
        return self._process_dataset(dataset, name, text_fields)

    def _load_dataset_with_config_safely(self, dataset_path: str, config: str, name: str, text_fields: Union[str, List[str]]) -> List[str]:
        """Load a dataset with config safely with error handling."""
        dataset = datasets.load_dataset(dataset_path, config)
        return self._process_dataset(dataset, name, text_fields)

    def load_medical_meadow(self) -> List[str]:
        """Load Medical Meadow datasets."""
        datasets_to_load = [
            ('Medical Flashcards', 'medalpaca/medical_meadow_medical_flashcards'),
            ('WikiDoc', 'medalpaca/medical_meadow_wikidoc'),
            ('WikiDoc Patient', 'medalpaca/medical_meadow_wikidoc_patient_information'),
            ('MediQA', 'medalpaca/medical_meadow_mediqa')
        ]
        
        all_texts = []
        for name, path in datasets_to_load:
            texts = self._load_dataset_safely(path, name, 'output')
            all_texts.extend(texts)
                
        return all_texts

    def load_medquad(self) -> List[str]:
        """Load MedQuAD dataset."""
        return self._load_dataset_safely('lavita/MedQuAD', 'MedQuAD', 'answer')

    def load_healthcare_magic(self) -> List[str]:
        """Load ChatDoctor-HealthCareMagic-100k dataset."""
        return self._load_dataset_safely('lavita/ChatDoctor-HealthCareMagic-100k', 'HealthCareMagic', 'output')

    def load_icliniq(self) -> List[str]:
        """Load ChatDoctor-iCliniq dataset (human answers only)."""
        return self._load_dataset_safely('lavita/ChatDoctor-iCliniq', 'iCliniq', ['answer_icliniq', 'answer_chatgpt', 'answer_chatdoctor'])

    def load_pubmed_qa(self) -> List[str]:
        """Load PubMedQA dataset using all available configs."""
        configs = ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled']
        all_texts = []
        
        for config in configs:
            texts = self._load_dataset_with_config_safely('qiaojin/PubMedQA', config, f'PubMedQA-{config}', 'long_answer')
            all_texts.extend(texts)
                
        return all_texts

    def load_medication_qa(self) -> List[str]:
        """Load MedicationQA dataset."""
        return self._load_dataset_safely('truehealth/medicationqa', 'MedicationQA', 'Answer')

    def load_liveqa(self) -> List[str]:
        """Load LiveQA dataset."""
        return self._load_dataset_safely('truehealth/liveqa', 'LiveQA', 'answer')

    def load_mental_health_counseling_conversations(self) -> List[str]:
        """Load Mental Health Counseling Conversations dataset."""
        return self._load_dataset_safely('amod/mental_health_counseling_conversations', 'Mental Health Counseling Conversations', 'Response')

    def load_medrag_pubmed(self) -> List[str]:
        """Load MedRAG PubMed dataset."""
        return self._load_dataset_safely('MedRAG/pubmed', 'MedRAG-PubMed', 'content')

    def load_medrag_textbooks(self) -> List[str]:
        """Load MedRAG textbooks dataset."""
        return self._load_dataset_safely('MedRAG/textbooks', 'MedRAG-Textbooks', 'content')

    def load_mayo_clinic(self) -> List[str]:
        """Load Mayo Clinic Symptoms and Diseases dataset."""
        return self._load_dataset_safely('celikmus/mayo_clinic_symptoms_and_diseases_v1', 'Mayo-Clinic', 'text')

    def load_bioinstruct(self) -> List[str]:
        """Load BioInstruct dataset."""
        return self._load_dataset_safely('bio-nlp-umass/bioinstruct', 'BioInstruct', 'output')
        
    def load_reddit_qa(self) -> List[str]:
        """Load RedditQA dataset."""
        return self._load_dataset_safely('DNivalis/reddit-health-qa', 'RedditQA', ['answer_1', 'answer_2', 'answer_3', 'answer_4', 'answer_5'])

    def _get_dataset_loaders(self) -> List[callable]:
        """Get all dataset loader functions."""
        return [
            self.load_medquad,
            self.load_medical_meadow,
            self.load_liveqa,
            self.load_healthcare_magic,
            self.load_icliniq,
            # self.load_pubmed_qa,
            self.load_medication_qa,
            self.load_mental_health_counseling_conversations,
            # self.load_medrag_pubmed,
            self.load_medrag_textbooks,
            self.load_mayo_clinic,
            self.load_bioinstruct,
            self.load_reddit_qa,
        ]

    def build_corpus(self) -> List[str]:
        """Build complete corpus and return texts with statistics."""
        all_texts = []
        loaders = self._get_dataset_loaders()
        
        for loader_func in loaders:
            texts = loader_func()
            logger.info(f"Loaded {len(texts)} texts from {loader_func.__name__}")
            all_texts.extend(texts)
        
        # Remove duplicates while preserving order
        unique_texts = list(dict.fromkeys(all_texts))
        logger.info(f"\nTotal unique texts in corpus: {len(unique_texts)}")
        
        return unique_texts


def _calculate_length_distribution(word_lengths: List[int]) -> Dict[str, int]:
    """Calculate length distribution statistics."""
    return {
        'short (< 50 words)': sum(1 for length in word_lengths if length < 50),
        'medium (50-200 words)': sum(1 for length in word_lengths if 50 <= length <= 200),
        'long (> 200 words)': sum(1 for length in word_lengths if length > 200)
    }


def _get_sample_texts(texts: List[str]) -> Dict[str, str]:
    """Get sample texts (shortest, median, longest)."""
    sorted_texts = sorted(texts, key=lambda x: len(x.split()))
    median_index = len(sorted_texts) // 2
    
    return {
        'shortest': sorted_texts[0],
        'median': sorted_texts[median_index],
        'longest': sorted_texts[-1]
    }


def analyze_corpus(texts: List[str]) -> Dict:
    """Analyze corpus statistics."""
    if not texts:
        return {
            'total_texts': 0,
            'avg_length': 0,
            'min_length': 0,
            'max_length': 0,
            'median_length': 0,
            'length_distribution': {
                'short (< 50 words)': 0,
                'medium (50-200 words)': 0,
                'long (> 200 words)': 0
            },
            'sample_texts': {
                'shortest': '',
                'median': '',
                'longest': ''
            }
        }
        
    word_lengths = [len(text.split()) for text in texts]
    sorted_lengths = sorted(word_lengths)
    median_index = len(sorted_lengths) // 2
    
    return {
        'total_texts': len(texts),
        'avg_length': sum(word_lengths) / len(texts),
        'min_length': min(word_lengths),
        'max_length': max(word_lengths),
        'median_length': sorted_lengths[median_index],
        'length_distribution': _calculate_length_distribution(word_lengths),
        'sample_texts': _get_sample_texts(texts)
    }