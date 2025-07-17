from pathlib import Path
from functools import lru_cache
from collections import Counter
import cmudict
import pyphen
from tqdm import tqdm

import numpy as np
import spacy

from bs4 import BeautifulSoup
from textstat import syllable_count

from utils.helpers import setup_logging

logger = setup_logging()


class TextClassifier:
    def __init__(self, model_name='en_core_web_trf', dale_chall_path=None, spache_path=None):
        """Initialize the TextClassifier with a spaCy model and word list paths."""        
        spacy.prefer_gpu()  # Use GPU if available

        # Load spaCy model directly
        self.nlp = spacy.load(model_name, disable={'ner'})
        
        # Load word lists for readability metrics
        self.dale_chall_easy_words = self._load_word_list(dale_chall_path)
        self.spache_familiar_words = self._load_word_list(spache_path)
        self.easy_words = self.dale_chall_easy_words | self.spache_familiar_words
        
        # Initialize syllable counting tools
        self.cmu_dict = cmudict.dict()
        self.pyphen = pyphen.Pyphen(lang='en_US')

    def _load_word_list(self, file_path):
        """Load a word list from a file with error handling."""
        words = set()
        if file_path is None:
            return words
            
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Warning: Word list file not found at {file_path}")
            return words
            
        words = {line.strip().lower() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}
        logger.info(f"Loaded {len(words)} words from {file_path}")
        return words

    def remove_html_tags(self, text):
        """Remove HTML/XML tags from text using BeautifulSoup."""
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")

    def normalize_whitespace(self, text):
        """Normalize whitespace by replacing multiple spaces with a single space."""
        return " ".join(text.split())

    def preprocess(self, text):
        """Preprocess text by cleaning and normalizing it."""
        text = self.remove_html_tags(text)
        text = self.normalize_whitespace(text)
        text = "".join(char for char in text if char.isprintable())
        return text

    def count_syllables(self, word):
        """Count syllables in a word using CMU dictionary or Pyphen fallback."""
        if not word:
            return 0
        
        if word in self.cmu_dict:
            cmu_phones = self.cmu_dict[word][0]
            return sum(1 for phoneme in cmu_phones if phoneme[-1].isdigit())
        
        return max(1, len(self.pyphen.positions(word)) + 1)

    def extract_statistics(self, texts):
        """Extract statistics from a batch of texts using spaCy's pipe with progress tracking."""
        preprocessed_texts = [self.preprocess(text) for text in texts]
        docs = list(tqdm(self.nlp.pipe(preprocessed_texts), total=len(preprocessed_texts), desc="Processing texts"))
        
        stats_list = []
        
        for doc in docs:
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            word_tokens = [token for token in doc if not token.is_punct and not token.is_space]
            words = [token.text for token in word_tokens]
            
            num_words = len(word_tokens)
            num_sentences = len(sentences)
            
            sentence_lengths = [sum(1 for token in sent if not token.is_punct and not token.is_space) 
                              for sent in doc.sents if sent.text.strip()]
            sentence_length_variation = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            syllable_counts = [self.count_syllables(token.text.lower()) for token in word_tokens]
            total_syllable_count = sum(syllable_counts)
            
            letter_count = sum(1 for token in word_tokens for char in token.text if char.isalpha())
            char_count = sum(len(token.text) for token in word_tokens)
            
            polysyllabic_count = sum(1 for count in syllable_counts if count >= 3)
            monosyllabic_count = sum(1 for count in syllable_counts if count == 1)
            mini_words_count = sum(1 for word in words if len(word) <= 3)
            
            unique_words = set(token.lemma_.lower() for token in word_tokens)
            
            stats = {
                "sentences": sentences,
                "word_tokens": word_tokens,
                "words": words,
                "syllable_counts": syllable_counts,
                "total_syllable_count": total_syllable_count,
                "letter_count": letter_count,
                "char_count": char_count,
                "polysyllabic_count": polysyllabic_count,
                "monosyllabic_count": monosyllabic_count,
                "mini_words_count": mini_words_count,
                "word_count": num_words,
                "sentence_count": num_sentences,
                "avg_word_length": char_count / max(1, num_words),
                "avg_syllables_per_word": total_syllable_count / max(1, num_words),
                "percent_polysyllabic_words": (polysyllabic_count / max(1, num_words)) * 100,
                "percent_monosyllabic_words": (monosyllabic_count / max(1, num_words)) * 100,
                "avg_sentence_length": num_words / max(1, num_sentences),
                "sentence_length_variation": sentence_length_variation,
                "type_token_ratio": len(unique_words) / max(1, num_words),
            }
            stats_list.append(stats)
        
        return stats_list

    def calculate_mtld_one_direction(self, words, threshold=0.72):
        """Calculate MTLD in one direction based on TTR threshold."""
        if not words:
            return 0.0
        
        segments = []
        current_segment_words = []
        unique_words = set()
        
        for word in words:
            current_segment_words.append(word)
            unique_words.add(word)
            ttr = len(unique_words) / len(current_segment_words)
            if ttr < threshold:
                segments.append(len(current_segment_words))
                current_segment_words = []
                unique_words = set()
        
        if current_segment_words:
            segments.append(len(current_segment_words))
        
        return sum(segments) / len(segments) if segments else 0.0

    def mtld(self, words, threshold=0.72):
        """Calculate Measure of Textual Lexical Diversity (MTLD)."""
        if not words:
            return 0.0
        
        words_lower = [word.lower() for word in words]
        mtld_forward = self.calculate_mtld_one_direction(words_lower, threshold)
        mtld_backward = self.calculate_mtld_one_direction(words_lower[::-1], threshold)
        
        return (mtld_forward + mtld_backward) / 2

    ### Readability Metric Methods

    def flesch_reading_ease(self, stats):
        """Calculate Flesch Reading Ease score."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        num_syllables = stats["total_syllable_count"]
        if num_words == 0 or num_sentences == 0:
            return 0.0
        return 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)

    def flesch_kincaid_grade(self, stats):
        """Calculate Flesch-Kincaid Grade Level."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        num_syllables = stats["total_syllable_count"]
        if num_words == 0 or num_sentences == 0:
            return 0.0
        return 0.39 * (num_words / num_sentences) + 11.8 * (num_syllables / num_words) - 15.59

    def smog_index(self, stats):
        """Calculate SMOG Index."""
        num_sentences = stats["sentence_count"]
        polysyllabic_count = stats["polysyllabic_count"]
        if num_sentences == 0:
            return 0.0
        return 1.043 * (polysyllabic_count * (30 / num_sentences)) ** 0.5 + 3.1291

    def coleman_liau_index(self, stats):
        """Calculate Coleman-Liau Index."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        num_letters = stats["letter_count"]
        if num_words == 0:
            return 0.0
        letters_per_100_words = num_letters / num_words * 100
        sentences_per_100_words = num_sentences / num_words * 100
        return 0.0588 * letters_per_100_words - 0.296 * sentences_per_100_words - 15.8

    def automated_readability_index(self, stats):
        """Calculate Automated Readability Index (ARI)."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        char_count = stats["char_count"]
        if num_words == 0 or num_sentences == 0:
            return 0.0
        return 4.71 * (char_count / num_words) + 0.5 * (num_words / num_sentences) - 21.43

    def dale_chall(self, stats):
        """Calculate Dale-Chall Readability Score."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        if num_words == 0 or num_sentences == 0:
            return 0.0
        
        difficult_words = self.count_difficult_words(stats)
        percent_difficult = (difficult_words / num_words) * 100
        score = 0.1579 * percent_difficult + 0.0496 * (num_words / num_sentences)
        if percent_difficult > 5:
            score += 3.6365
        return score

    def count_difficult_words(self, stats):
        """Count difficult words not in Dale-Chall easy word list."""
        return sum(1 for token in stats["word_tokens"] 
                    if token.text.lower() not in self.easy_words 
                    and token.lemma_.lower() not in self.easy_words)

    def linsear_write_formula(self, stats):
        """Calculate Linsear Write Formula."""
        num_sentences = stats["sentence_count"]
        if num_sentences == 0:
            return 0.0
        easy_words = sum(1 for count in stats["syllable_counts"] if count <= 2)
        hard_words = stats["polysyllabic_count"]
        score = (easy_words + 3 * hard_words) / num_sentences
        if score > 20:
            return score / 2
        return (score / 2) - 2

    def gunning_fog(self, stats):
        """Calculate Gunning Fog Index."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        complex_words = stats["polysyllabic_count"]
        if num_words == 0 or num_sentences == 0:
            return 0.0
        return 0.4 * ((num_words / num_sentences) + 100 * (complex_words / num_words))

    def text_standard(self, stats):
        """Calculate Text Standard using consensus of grade levels."""
        fk_grade = self.flesch_kincaid_grade(stats)
        fog_grade = self.gunning_fog(stats)
        smog_grade = self.smog_index(stats)
        cl_grade = self.coleman_liau_index(stats)
        ari_grade = self.automated_readability_index(stats)
        lw_grade = self.linsear_write_formula(stats)
        fre_score = self.flesch_reading_ease(stats)

        fre_grade = 13
        if fre_score >= 90:
            fre_grade = 5
        elif fre_score >= 80:
            fre_grade = 6
        elif fre_score >= 70:
            fre_grade = 7
        elif fre_score >= 60:
            fre_grade = 8.5
        elif fre_score >= 50:
            fre_grade = 10
        elif fre_score >= 40:
            fre_grade = 11
        elif fre_score >= 30:
            fre_grade = 12
        
        grades = [int(grade) for grade in [fk_grade, fog_grade, smog_grade, cl_grade, ari_grade, lw_grade] if grade > 0]
        grades.extend([int(grade) + 1 for grade in [fk_grade, fog_grade, smog_grade, cl_grade, ari_grade, lw_grade] if grade > 0])
        grades.append(int(fre_grade))
        
        if not grades:
            return 0.0
        
        counter = Counter(grades)
        return float(counter.most_common(1)[0][0])

    def spache(self, stats):
        """Calculate Spache Readability Score."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        if num_words == 0 or num_sentences == 0:
            return 0.0
        
        unfamiliar_words = self.count_difficult_words(stats)
        percent_unfamiliar = (unfamiliar_words / num_words) * 100
        return 0.141 * (num_words / num_sentences) + 0.086 * percent_unfamiliar + 0.839

    def mcalpine_eflaw(self, stats):
        """Calculate McAlpine EFLAW Readability Score."""
        num_words = stats["word_count"]
        num_sentences = stats["sentence_count"]
        mini_words = stats["mini_words_count"]
        
        if num_sentences == 0:
            return 0.0
        
        return (num_words + mini_words) / num_sentences

    def forcast(self, stats):
        """Calculate FORCAST Readability Grade Level."""
        num_words = stats["word_count"]
        monosyllabic_count = stats["monosyllabic_count"]
        if num_words == 0:
            return 0.0
        scaled_monosyllabic = (monosyllabic_count / num_words) * 150
        grade_level = 20 - (scaled_monosyllabic / 10)
        return max(0.0, grade_level)

    def predict_single(self, text):
        """Predict readability metrics for a single text."""
        return self.predict_batch([text])[0]
    
    def predict_batch(self, texts, batch_size=32):
        """Calculate readability metrics for a batch of texts."""
        stats_list = self.extract_statistics(texts)
        results = []
        
        for stats in stats_list:
            metrics = {
                'flesch_reading_ease': self.flesch_reading_ease(stats),
                'flesch_kincaid_grade': self.flesch_kincaid_grade(stats),   
                'smog_index': self.smog_index(stats),
                'coleman_liau_index': self.coleman_liau_index(stats),
                'automated_readability_index': self.automated_readability_index(stats),
                'linsear_write_formula': self.linsear_write_formula(stats),
                'gunning_fog': self.gunning_fog(stats),
                'text_standard': self.text_standard(stats),
                'mcalpine_eflaw': self.mcalpine_eflaw(stats),
                'dale_chall': self.dale_chall(stats),
                'spache': self.spache(stats),
                'forcast': self.forcast(stats),
                'difficult_words_percent': self.count_difficult_words(stats) / max(1, stats["word_count"]) * 100, 
                'type_token_ratio': stats["type_token_ratio"],
                'avg_word_length': stats["avg_word_length"],
                'avg_syllables_per_word': stats["avg_syllables_per_word"],
                'percent_polysyllabic_words': stats["percent_polysyllabic_words"],
                'percent_monosyllabic_words': stats["percent_monosyllabic_words"],
                'avg_sentence_length': stats["avg_sentence_length"],
                'mtld': self.mtld(stats["words"])
            }
            results.append(metrics)
        
        return results

if __name__ == "__main__":
    classifier = TextClassifier()
    sample_text = """Intracranial neoplasms, despite histological benignity, warrant careful clinical consideration due to mass effect, location-dependent symptomatology, and potential for growth. Meningiomas, schwannomas, and pituitary adenomas comprise common benign variants. Clinical significance varies with tumor size, growth rate, anatomical location, and proximity to critical structures. Mass effect can precipitate increased intracranial pressure, focal neurological deficits, and seizure activity. Management approaches include surveillance, surgical resection, or radiation therapy, determined by factors including tumor characteristics, symptomatology, and patient-specific considerations. Five-year survival rates typically exceed 90% with appropriate intervention."""
    metrics = classifier.predict_single(sample_text)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")