import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Term:
    """Represents a biomedical term with its properties."""
    text: str
    score: float
    semantic_type: str

@dataclass
class ComplexityScores:
    """Contains all complexity metrics for analyzed text."""
    professional_terms_ratio: float
    core_professional_ratio: float
    avg_chv_familiarity: float
    distinct_semantic_types: int
    semantic_types: List[str]

class BiomedicalComplexityAnalyzer:
    """
    Analyzes complexity of biomedical text using various metrics.
    
    This analyzer uses MetaMapLite to extract medical terms and concepts,
    then calculates complexity scores based on professional terminology usage
    and semantic diversity.
    
    Attributes:
        VALID_SOURCES: Set of valid vocabulary sources for MetaMapLite
    """
    
    VALID_SOURCES = {'SNOMEDCT_US', 'CHV'}
    
    def __init__(self, metamaplite_path: str, chv_file: str):
        """
        Initialize the analyzer with required paths and data.
        
        Args:
            metamaplite_path: Path to MetaMapLite installation
            chv_file: Path to Consumer Health Vocabulary file
        
        Raises:
            FileNotFoundError: If required files are not found
            ValueError: If paths are invalid
        """
        self._validate_paths(metamaplite_path, chv_file)
        self.metamaplite_path = Path(metamaplite_path)
        self.chv_data = self._load_chv_data(chv_file)
        
    def _validate_paths(self, metamaplite_path: str, chv_file: str) -> None:
        """Validate that all required files and paths exist."""
        if not os.path.exists(metamaplite_path):
            raise FileNotFoundError(f"MetaMapLite path not found: {metamaplite_path}")
        if not os.path.exists(chv_file):
            raise FileNotFoundError(f"CHV file not found: {chv_file}")
            
        # Validate MetaMapLite directory structure
        required_files = [
            "target/metamaplite-3.6.2rc8-standalone.jar",
            "data/ivf/2022AB/USAbase",
            "data/models",
            "data/specialterms.txt",
            "config/metamaplite.properties"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(metamaplite_path, file)):
                raise FileNotFoundError(f"Required MetaMapLite file not found: {file}")

    def _load_chv_data(self, chv_file: str) -> Dict:
        """Load and preprocess CHV data with individual score components."""
        logger.info("Loading CHV data...")
        
        # Read necessary columns including individual scores
        df = pd.read_csv(
            chv_file,
            usecols=['CUI', 'Frequency Score', 'Context Score', 'CUI Score', 'Combo Score', 'CHV preferred'],
            na_values=['\\N', -1],
            dtype={
                'CUI': str,
                'Frequency Score': float,
                'Context Score': float,
                'CUI Score': float,
                'Combo Score': float,
                'CHV preferred': str
            }
        )
        
        # Calculate mean scores for each component
        self.mean_freq_score = df['Frequency Score'].mean()
        self.mean_context_score = df['Context Score'].mean()
        self.mean_cui_score = df['CUI Score'].mean()
        
        # Create dictionary with component scores
        cui_data = {}
        new_combo_scores = []  # To collect all new combo scores for mean calculation
        
        for cui, freq, context, cui_score, combo, preferred in zip(
            df['CUI'],
            df['Frequency Score'],
            df['Context Score'],
            df['CUI Score'],
            df['Combo Score'],
            df['CHV preferred'] == 'yes'
        ):
            # Fill missing individual scores with means
            freq = freq if pd.notna(freq) else self.mean_freq_score
            context = context if pd.notna(context) else self.mean_context_score
            cui_score = cui_score if pd.notna(cui_score) else self.mean_cui_score
            
            # Recalculate combo score as average of available components
            available_scores = [s for s in [freq, context, cui_score] if pd.notna(s)]
            new_combo = sum(available_scores) / len(available_scores) if available_scores else 0
            new_combo_scores.append(new_combo)
            
            cui_data[cui] = {
                'freq_score': freq,
                'context_score': context,
                'cui_score': cui_score,
                'score': new_combo,
                'preferred': preferred
            }
        
        # Calculate mean_score after all new combo scores are calculated
        self.mean_score = np.mean(new_combo_scores)
        
        logger.info(f"Loaded {len(cui_data)} CHV terms")
        return cui_data

    def analyze_complexity(self, text: str) -> ComplexityScores:
        """
        Analyze text complexity using multiple metrics.
        
        Args:
            text: The biomedical text to analyze
            
        Returns:
            ComplexityScores object containing all metrics
        """
        if not text.strip():
            return self._create_empty_scores()
            
        # Extract terms from different vocabularies in parallel
        terms = {
            'umls': self._extract_terms(self.run_metamaplite(text)),
            'snomed': self._extract_terms(self.run_metamaplite(text, "SNOMEDCT_US")),
            'chv': self._extract_terms(self.run_metamaplite(text, "CHV"))
        }
        
        return self._calculate_all_scores(terms)

    def run_metamaplite(self, text: str, source: Optional[str] = None) -> str:
        """
        Run MetaMapLite with optimized settings.
        
        Args:
            text: Text to analyze
            source: Optional vocabulary source restriction
            
        Returns:
            MetaMapLite output as string
        """
        if source and source not in self.VALID_SOURCES:
            raise ValueError(f"Invalid source: {source}. Must be one of {self.VALID_SOURCES}")
            
        cmd = [
            "java",
            "-Xmx4g",  # Increase heap size for better performance
            "-cp", str(self.metamaplite_path / "target/metamaplite-3.6.2rc8-standalone.jar"),
            "gov.nih.nlm.nls.ner.MetaMapLite",
            f"--indexdir={self.metamaplite_path}/data/ivf/2022AB/USAbase",
            f"--modelsdir={self.metamaplite_path}/data/models",
            f"--specialtermsfile={self.metamaplite_path}/data/specialterms.txt",
            f"--configfile={self.metamaplite_path}/config/metamaplite.properties",
            "--pipe"
        ]
        
        if source:
            cmd.append(f"--restrict_to_sources={source}")
            
        try:
            process = subprocess.run(
                cmd,
                input=text,
                text=True,
                capture_output=True,
                check=True
            )
            return process.stdout
        except subprocess.CalledProcessError as e:
            logger.error(f"MetaMapLite error: {e.stderr}")
            raise RuntimeError("MetaMapLite processing failed") from e

    def _extract_terms(self, metamap_output: str) -> Dict[str, Term]:
        """Extract and parse terms from MetaMapLite output efficiently."""
        terms = {}
        for line in metamap_output.splitlines():
            if not line.strip():
                continue
                
            try:
                parts = line.strip().split('|')
                if len(parts) >= 7:
                    terms[parts[4]] = Term(
                        text=parts[3].lower(),
                        score=float(parts[2]),
                        semantic_type=parts[5].strip('[]')
                    )
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing line: {line}. Error: {e}")
                
        return terms

    def _calculate_all_scores(self, terms: Dict[str, Dict[str, Term]]) -> ComplexityScores:
        """Calculate all complexity metrics efficiently."""
        n_health_terms = len(terms['umls'])
        
        if n_health_terms == 0:
            return self._create_empty_scores()
            
        # Calculate term-level scores
        prof_ratio = len(terms['snomed']) / n_health_terms
        core_professional = set(terms['snomed'].keys()) - set(terms['chv'].keys())
        core_ratio = len(core_professional) / n_health_terms
        
        # Calculate CHV familiarity
        chv_scores = [
            self.chv_data.get(cui, {'score': self.mean_score})['score']
            for cui in terms['chv']
        ]
        avg_familiarity = np.mean(chv_scores) if chv_scores else 0
        
        # Calculate semantic scores
        semantic_types = {term.semantic_type for term in terms['umls'].values()}
        
        return ComplexityScores(
            professional_terms_ratio=prof_ratio,
            core_professional_ratio=core_ratio,
            avg_chv_familiarity=avg_familiarity,
            distinct_semantic_types=len(semantic_types),
            semantic_types=list(semantic_types)
        )

    def _create_empty_scores(self) -> ComplexityScores:
        """Create empty scores for invalid input."""
        return ComplexityScores(
            professional_terms_ratio=0,
            core_professional_ratio=0,
            avg_chv_familiarity=0,
            distinct_semantic_types=0,
            semantic_types=[]
        )

def main():
    """Example usage of the BiomedicalComplexityAnalyzer."""
    analyzer = BiomedicalComplexityAnalyzer(
        metamaplite_path="D:/CHELC/public_mm_lite_3.6.2rc8",
        chv_file="D:/CHELC/CHV_concepts_terms_flatfile_20110204.csv"
    )

    test_cases = [
        "The patient has a headache.",
        "The patient presents with severe myocardial infarction and hypertension.",
        "Patient exhibits acute exacerbation of chronic obstructive pulmonary disease.",
        "Basic metabolic panel shows hyperkalemia and elevated creatinine.",
        "Diabetes is a condition that happens when there is too much sugar in your blood. The sugar in your blood, called glucose, comes from the food you eat and gives your body energy. Insulin is a hormone made by the pancreas. It helps move glucose from your blood into your cells, where it's used for energy.",
        "Diabetes mellitus is a metabolic disorder resulting from chronic hyperglycemia due to defects in insulin secretion, insulin action, or both. Blood glucose, primarily obtained from dietary intake, is essential for cellular respiration. Insulin, synthesized by the B-cells of the pancreatic islets, mediates cellular glucose uptake via GLUT transporters. In diabetes, insufficient insulin production or peripheral insulin resistance disrupts glucose homeostasis, leading to pathological sequelae such as microvascular and macrovascular complications."
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: '{text}'")
        results = analyzer.analyze_complexity(text)
        
        print("\nTerm Level Complexity:")
        print(f"  Professional Terms Ratio: {results.professional_terms_ratio:.2f}")
        print(f"  Core Professional Ratio: {results.core_professional_ratio:.2f}")
        print(f"  Average CHV Familiarity: {results.avg_chv_familiarity:.2f}")

        print("\nSemantic Level Complexity:")
        print(f"  Distinct Semantic Types: {results.distinct_semantic_types}")
        print(f"  Semantic Types: {', '.join(results.semantic_types)}")

if __name__ == "__main__":
    main()