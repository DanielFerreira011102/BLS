import os
import networkx
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import subprocess
import logging
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Term:
    """Represents a biomedical term with its properties."""
    text: str
    score: float
    semantic_type: str
    cui: str = ''  # Default empty string for CUI
    ancestors: Set[str] = None  # Will be initialized in post_init
    length: int = 0  # Will be initialized in post_init
    
    def __post_init__(self):
        if self.ancestors is None:
            self.ancestors = set()
        if self.length == 0:
            self.length = len(self.text.split())
            
@dataclass
class ComplexityScores:
    """Contains all complexity metrics for analyzing biomedical text complexity."""
    
    # Term-level metrics
    professional_terms_ratio: float
    """Ratio of professional medical terms to all recognized health terms (0.0 to 1.0).
    
    Measures how many of the identified health terms appear in SNOMED-CT (professional
    medical terminology) compared to all health terms found in UMLS.
    
    Calculation:
    1. Numerator: Number of terms found in SNOMED-CT
    2. Denominator: Total number of health terms found in UMLS
    3. Formula: len(terms['snomed']) / len(terms['umls'])
    
    Interpretation:
    - 1.0: All health terms are professional medical terms (SNOMED-CT)
    - 0.0: No professional medical terms found
    - 0.5: Half of the health terms are professional medical terms
    """
    
    core_professional_ratio: float
    """Ratio of exclusively professional medical terms (0.0 to 1.0).
    
    Measures terms that appear only in professional vocabulary (SNOMED-CT) and not in
    consumer health vocabulary (CHV), indicating terminology unlikely to be familiar
    to general consumers.
    
    Calculation:
    1. Numerator: Number of terms in SNOMED-CT but not in CHV
    2. Denominator: Total number of health terms found in UMLS
    3. Formula: len(set(terms['snomed']) - set(terms['chv'])) / len(terms['umls'])
    
    Interpretation:
    - 1.0: All health terms are exclusively professional (no consumer equivalents)
    - 0.0: All health terms have consumer-friendly alternatives
    - 0.5: Half of the terms are exclusively professional
    """
    
    avg_chv_familiarity: float
    """Average consumer familiarity score for health terms (0.0 to 1.0).
    
    Measures how familiar terms are to general consumers based on three CHV scores:
    1. Frequency Score: How often the term appears in consumer health texts
    2. Context Score: How difficult the term is based on its usage context
    3. CUI Score: Term difficulty based on relationship to known easy/hard concepts
    
    Calculation:
    1. For each term found in CHV:
       - Get individual scores (frequency, context, CUI)
       - Fill missing scores with mean values for that score type
       - Calculate term's combo score as average of available scores
    2. Final score is average of all terms' combo scores
    
    Interpretation:
    - 1.0: Terms very familiar to general consumers
    - 0.0: Terms very unfamiliar to general consumers
    - 0.5: Moderate consumer familiarity
    """
    
    specialized_term_ratio: float
    """Ratio of medical terms to total words in text (0.0 to 1.0).
    
    Measures the density of medical terminology in the text, regardless of whether
    terms are professional or consumer-friendly.
    
    Calculation:
    1. Numerator: Number of recognized health terms (UMLS)
    2. Denominator: Total number of words in text
    3. Formula: len(terms['umls']) / len(word_tokenize(text))
    
    Interpretation:
    - 1.0: Every word is a medical term
    - 0.0: No medical terms
    - Higher values indicate more concentrated medical content
    """
    
    avg_term_length: float
    """Average number of words in medical terms (≥1.0).
    
    Measures the structural complexity of medical terms by their word count,
    capturing the prevalence of compound medical terms.
    
    Calculation:
    1. For each term in UMLS:
       - Count words: len(term.text.split())
    2. Calculate mean of all term lengths
    
    Interpretation:
    - 1.0: All terms are single words
    - >1.0: Average length of compound terms
    - Higher values indicate more complex, multi-word terminology
    """
    
    multi_word_ratio: float
    """Proportion of medical terms containing multiple words (0.0 to 1.0).
    
    Measures how many medical terms are compound terms rather than single words,
    indicating terminological complexity.
    
    Calculation:
    1. Numerator: Number of terms with more than one word
    2. Denominator: Total number of terms
    3. Formula: len([t for t in terms if len(t.text.split()) > 1]) / len(terms)
    
    Interpretation:
    - 1.0: All terms are multi-word
    - 0.0: All terms are single words
    - Higher values indicate more complex terminology
    """
    
    tfidf_complexity: float
    """Term Frequency-Inverse Document Frequency based measure of term specificity (0.0 to 1.0).
    
    Measures how specific or unique terms are in the text using TF-IDF scores,
    calculated over n-grams (1-3 words) to capture multi-word medical terms.
    
    Calculation:
    1. Generate n-grams (1-3 words) from text
    2. Calculate TF-IDF matrix using sklearn's TfidfVectorizer
       - Term Frequency: Normalized count of term appearances
       - IDF: Log of (1 + total n-grams / n-grams containing term)
    3. Take mean of TF-IDF matrix values
    
    Interpretation:
    - Higher values: More specific/technical terms used repeatedly
    - Lower values: More general/varied terminology
    - Accounts for both term specificity and frequency
    """

    # Semantic-level metrics
    distinct_semantic_types: int
    """Number of unique UMLS semantic types found in text (≥0).
    
    Counts how many different categories of medical concepts appear in the text,
    based on UMLS semantic network categories.
    
    Calculation:
    1. Extract semantic type for each term from UMLS
    2. Count unique semantic types
    3. Formula: len(set(term.semantic_type for term in terms))
    
    Interpretation:
    - Higher values: Text covers more diverse medical concepts
    - Lower values: Text focuses on fewer concept types
    """
    
    semantic_types: List[str]
    """List of UMLS semantic types present in text.
    
    Provides qualitative insight into the categories of medical concepts discussed.
    Each type is a standardized UMLS semantic network category.
    
    Structure:
    - List of unique semantic type abbreviations
    - Standard UMLS semantic type codes
    """
    
    concept_density: float
    """Average number of medical concepts per sentence (≥0.0).
    
    Measures how many distinct medical concepts (UMLS terms) appear per sentence,
    indicating conceptual density of the text.
    
    Calculation:
    1. Numerator: Total number of UMLS concepts found
    2. Denominator: Number of sentences
    3. Formula: len(terms['umls']) / len(sent_tokenize(text))
    
    Interpretation:
    - Higher values: More medical concepts per sentence
    - Lower values: Fewer medical concepts per sentence
    """
    
    semantic_similarity: float
    """Average pairwise semantic similarity between sentences (0.0 to 1.0).
    
    Measures how conceptually related the sentences are using biomedical embeddings,
    indicating text coherence and topic consistency.
    
    Calculation:
    1. Encode each sentence using biomedical language model
    2. Normalize embeddings to unit length
    3. Calculate pairwise cosine similarities between sentence embeddings
    4. Take mean of upper triangle of similarity matrix (unique pairs)
    
    Interpretation:
    - 1.0: Sentences are semantically identical
    - ~0.7: Highly related sentences (same topic, different aspects)
    - ~0.3: Moderately related sentences (related topics)
    - ~0.1: Weakly related sentences (different topics)
    - 0.0: Completely unrelated sentences
    
    Note: Single sentences return 1.0 as they are identical to themselves.
    """
    
    hierarchical_depth: float
    """Average depth of medical concepts in UMLS hierarchy (≥0.0).
    
    Measures how specific/specialized the medical concepts are by their position
    in the UMLS concept hierarchy.
    
    Calculation:
    1. For each term, find all ancestor concepts in UMLS
    2. Calculate shortest path lengths to root concepts
    3. Take mean of all path lengths
    
    Interpretation:
    - Higher values: More specific/specialized concepts
    - Lower values: More general concepts
    """
    
    semantic_coherence: float
    """Clustering coefficient of concept similarity graph (0.0 to 1.0).
    
    Measures how interconnected the medical concepts are based on their
    semantic similarities, indicating conceptual cohesion.
    
    Calculation:
    1. Build graph where:
       - Nodes are medical concepts
       - Edges connect concepts with similarity > 0.7
    2. Calculate average clustering coefficient
    
    Interpretation:
    - 1.0: All concepts highly related
    - 0.0: No concept relationships
    - Higher values indicate more interconnected concepts
    """
    
    semantic_network_density: float
    """Density of connections in concept similarity graph (0.0 to 1.0).
    
    Measures how many semantic relationships exist between concepts compared
    to the maximum possible relationships.
    
    Calculation:
    1. Using same concept graph as semantic_coherence
    2. Formula: (2 * edges) / (nodes * (nodes - 1))
    
    Interpretation:
    - 1.0: All concepts related to all others
    - 0.0: No relationships between concepts
    - Higher values indicate more interconnected terminology
    """

class BiomedicalComplexityAnalyzer:
    VALID_SOURCES = {'SNOMEDCT_US', 'CHV'}
    RELEVANT_SEMANTIC_TYPES = set()  # Add your 84 relevant types here
    
    def __init__(self, metamaplite_path: str, chv_file: str, umls_path: str):
        self._validate_paths(metamaplite_path, chv_file, umls_path)
        self.metamaplite_path = Path(metamaplite_path)
        self.chv_data = self._load_chv_data(chv_file)
        self.umls_graph = self._load_umls_graph(umls_path)
        self.bio_embeddings = SentenceTransformer('gsarti/biobert-nli')
        self.tfidf = TfidfVectorizer(ngram_range=(1, 3))

    def _validate_paths(self, metamaplite_path: str, chv_file: str, umls_path: str):
        """Validate that all required files and paths exist."""
        if not os.path.exists(metamaplite_path):
            raise FileNotFoundError(f"MetaMapLite path not found: {metamaplite_path}")
        if not os.path.exists(chv_file):
            raise FileNotFoundError(f"CHV file not found: {chv_file}")    
        # if not os.path.exists(umls_path):
        #     raise FileNotFoundError(f"UMLS path not found: {umls_path}")
        
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
    
    def _load_umls_graph(self, umls_path: str) -> nx.DiGraph:
        """Load UMLS concept hierarchy into a directed graph."""
        G = nx.DiGraph()
        # Load UMLS relationships and build graph
        # This is a placeholder - implement actual UMLS loading logic
        return G
        
    def analyze_complexity(self, text: str) -> ComplexityScores:
        """Analyze text complexity using multiple metrics."""
        if not text.strip():
            return self._create_empty_scores()
            
        # Extract terms and concepts
        terms = self._extract_all_terms(text)
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Calculate term-level complexity
        term_scores = self._calculate_term_complexity(terms, words, text)
        
        # Calculate semantic complexity
        semantic_scores = self._calculate_semantic_complexity(terms, sentences, text)
        
        return ComplexityScores(**{**term_scores, **semantic_scores})
    
    def _calculate_term_complexity(self, terms: Dict[str, Term], words: List[str], text: str) -> Dict:
        """Calculate comprehensive term-level complexity metrics."""
        n_health_terms = len(terms['umls'])
        if n_health_terms == 0:
            return self._create_empty_term_scores()
        
        # Basic ratios
        prof_ratio = len(terms['snomed']) / n_health_terms
        core_professional = set(terms['snomed'].keys()) - set(terms['chv'].keys())
        core_ratio = len(core_professional) / n_health_terms
        
        # CHV familiarity with component scores
        chv_scores = [
            self.chv_data.get(cui, {'score': self.mean_score})['score']
            for cui in terms['chv']
        ]
        avg_familiarity = np.mean(chv_scores) if chv_scores else 0
        
        # Additional term complexity metrics
        specialized_ratio = n_health_terms / len(words)
        term_lengths = [term.length for term in terms['umls'].values()]
        avg_term_length = np.mean(term_lengths) if term_lengths else 0
        multi_word_ratio = len([t for t in terms['umls'].values() if t.length > 1]) / n_health_terms
        
        # TF-IDF based complexity
        tfidf_matrix = self.tfidf.fit_transform([text])
        tfidf_complexity = float(tfidf_matrix.mean())
        
        return {
            'professional_terms_ratio': prof_ratio,
            'core_professional_ratio': core_ratio,
            'avg_chv_familiarity': avg_familiarity,
            'specialized_term_ratio': specialized_ratio,
            'avg_term_length': avg_term_length,
            'multi_word_ratio': multi_word_ratio,
            'tfidf_complexity': tfidf_complexity
        }
        
    def _calculate_semantic_complexity(self, terms: Dict[str, Term], sentences: List[str], text: str) -> Dict:
        """Calculate semantic-level complexity metrics with safe graph handling."""
        if not terms['umls']:
            return self._create_empty_semantic_scores()
                
        # Basic semantic type analysis
        semantic_types = {term.semantic_type for term in terms['umls'].values()}
        
        # Concept density
        concept_density = len(terms['umls']) / len(sentences)
        
        # Semantic similarity using biomedical embeddings
        if len(sentences) > 1:
            embeddings = self.bio_embeddings.encode(sentences)
            # Normalize embeddings for cosine similarity
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarity_matrix = np.inner(embeddings, embeddings)
            # Take upper triangle only (excluding diagonal) to avoid counting same pairs twice
            upper_triangle = np.triu(similarity_matrix, k=1)
            semantic_similarity = float(upper_triangle[upper_triangle != 0].mean())
        else:
            semantic_similarity = 1.0
        
        # Safe graph-based calculations
        try:
            depths = []
            for term in terms['umls'].values():
                if term.cui in self.umls_graph:
                    paths = nx.shortest_path_length(self.umls_graph, term.cui)
                    depths.extend(paths.values())
            hierarchical_depth = np.mean(depths) if depths else 0
        except (networkx.exception.NetworkXError, KeyError):
            hierarchical_depth = 0
                
        # Build concept graph safely
        G = self._build_concept_graph(terms['umls'])
        semantic_coherence = nx.average_clustering(G) if G.nodes else 0
        semantic_network_density = nx.density(G)
        
        return {
            'distinct_semantic_types': len(semantic_types),
            'semantic_types': list(semantic_types),
            'concept_density': concept_density,
            'semantic_similarity': semantic_similarity,
            'hierarchical_depth': hierarchical_depth,
            'semantic_coherence': semantic_coherence,
            'semantic_network_density': semantic_network_density
        }
        
    def _build_concept_graph(self, terms: Dict[str, Term]) -> nx.Graph:
        """Build semantic network from concepts with efficient batch encoding."""
        G = nx.Graph()
        
        # Add nodes
        for cui, term in terms.items():
            G.add_node(cui, semantic_type=term.semantic_type)
        
        # Add edges based on semantic similarity
        if len(terms) > 1:
            # Encode all terms at once
            term_texts = [term.text for term in terms.values()]
            embeddings = self.bio_embeddings.encode(term_texts)
            
            # Calculate similarities using matrix operations
            similarity_matrix = np.inner(embeddings, embeddings)
            
            # Add edges where similarity > threshold
            term_cuis = list(terms.keys())
            for i in range(len(term_cuis)):
                for j in range(i + 1, len(term_cuis)):
                    similarity = float(similarity_matrix[i, j])
                    if similarity > 0.7:
                        G.add_edge(term_cuis[i], term_cuis[j], weight=similarity)
        
        return G

    def _extract_all_terms(self, text: str) -> Dict[str, Dict[str, Term]]:
        """Extract terms from all vocabularies with safe ancestor handling."""
        terms = {
            'umls': self._extract_terms(self.run_metamaplite(text)),
            'snomed': self._extract_terms(self.run_metamaplite(text, "SNOMEDCT_US")),
            'chv': self._extract_terms(self.run_metamaplite(text, "CHV"))
        }
        
        # Safely handle ancestors even with empty graph
        for source_terms in terms.values():
            for cui, term in source_terms.items():
                try:
                    term.ancestors = set(nx.ancestors(self.umls_graph, cui))
                except (networkx.exception.NetworkXError, KeyError):
                    term.ancestors = set()  # Empty set if node not in graph
                
        return terms

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
                    cui = parts[4]
                    terms[cui] = Term(
                        text=parts[3].lower(),
                        score=float(parts[2]),
                        semantic_type=parts[5].strip('[]'),
                        cui=cui
                    )
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing line: {line}. Error: {e}")
                
        return terms

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
        chv_file="D:/CHELC/CHV_concepts_terms_flatfile_20110204.csv",
        umls_path="D:/CHELC/umls2020AB/META"  # Placeholder path
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
        print(f"  Specialized Term Ratio: {results.specialized_term_ratio:.2f}")
        print(f"  Average Term Length: {results.avg_term_length:.2f}")
        print(f"  Multi-Word Ratio: {results.multi_word_ratio:.2f}")
        print(f"  TF-IDF Complexity: {results.tfidf_complexity:.2f}")

        print("\nSemantic Level Complexity:")
        print(f"  Distinct Semantic Types: {results.distinct_semantic_types}")
        print(f"  Semantic Types: {', '.join(results.semantic_types)}")
        print(f"  Concept Density: {results.concept_density:.2f}")
        print(f"  Semantic Similarity: {results.semantic_similarity:.2f}")
        print(f"  Hierarchical Depth: {results.hierarchical_depth:.2f}")
        print(f"  Semantic Coherence: {results.semantic_coherence:.2f}")
        print(f"  Semantic Network Density: {results.semantic_network_density:.2f}")

if __name__ == "__main__":
    main()