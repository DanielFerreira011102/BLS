import re
import time
import logging
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig

from utils.helpers import setup_logging

logger = setup_logging(logging.DEBUG)


class ReadabilityResponseParser:
    """Handles parsing of model responses for readability assessment in XML format."""
    
    # Define all tag names we need to handle
    TAG_NAMES = [
        "vocabulary_complexity",
        "syntactic_complexity",
        "conceptual_density",
        "background_knowledge",
        "cognitive_load",
        "reasoning"
    ]
    
    # Default values as constants
    DEFAULT_SCORE = 2.5
    DEFAULT_REASONING = "Failed to parse response"
    
    # Compile patterns for each tag
    TAG_PATTERNS = {}
    for tag in TAG_NAMES:
        TAG_PATTERNS[tag] = (
            re.compile(f"<{tag}>(.*?)</{tag}>", re.DOTALL),  # Standard pattern
            re.compile(f"<{tag}>(.*?)(?=<(?!{tag})|$)", re.DOTALL)  # Fallback pattern
        )
    
    @staticmethod
    def remove_thinking_section(text: str) -> str:
        """Remove thinking trace sections from text."""
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()
    
    @staticmethod
    def extract_content_between_tags(text: str, tag_name: str) -> Optional[str]:
        """Extract content between specific tags with fallback for malformed XML."""
        if not text or tag_name not in ReadabilityResponseParser.TAG_PATTERNS:
            return None
        
        # Get pre-compiled patterns for this tag
        pattern, fallback_pattern = ReadabilityResponseParser.TAG_PATTERNS[tag_name]
        
        # Try standard pattern first
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
        
        # Try fallback pattern
        fallback_match = fallback_pattern.search(text)
        if fallback_match:
            logger.warning(f"Using fallback extraction for unclosed <{tag_name}> tag")
            return fallback_match.group(1).strip()
        
        return None
    
    @staticmethod
    def get_default_scores() -> Dict[str, int]:
        """Return default dimension scores used when parsing fails."""
        dimension_tags = ReadabilityResponseParser.TAG_NAMES[:-1]  # All except reasoning
        return {tag: ReadabilityResponseParser.DEFAULT_SCORE for tag in dimension_tags}
    
    @staticmethod
    def get_default_response() -> Dict[str, Any]:
        """Return the complete default response structure."""
        return {
            "dimension_scores": ReadabilityResponseParser.get_default_scores(),
            "reasoning": ReadabilityResponseParser.DEFAULT_REASONING,
            "overall_score": ReadabilityResponseParser.DEFAULT_SCORE
        }
    
    @staticmethod
    def parse_readability_response(raw_response: str) -> Dict[str, Any]:
        """Parse readability assessment response."""
        # Handle empty response
        if not raw_response:
            logger.warning(f"Empty response received")
            return ReadabilityResponseParser.get_default_response()
        
        # Remove thinking section and work with the full text
        text = ReadabilityResponseParser.remove_thinking_section(raw_response)
        if not text:
            logger.warning(f"No content after removing thinking section: {raw_response}")
            return ReadabilityResponseParser.get_default_response()
        
        # Extract dimension scores
        dimension_scores = ReadabilityResponseParser.get_default_scores().copy()
        
        # Log raw response if any tag isn't found and parsed correctly
        for tag in dimension_scores.keys():
            value = ReadabilityResponseParser.extract_content_between_tags(text, tag)
            if not value or not value.isdigit():
                logger.warning(f"Unable to parse <{tag}> tag. Raw response: {raw_response}")
                break
            dimension_scores[tag] = int(value)
        
        # Extract reasoning
        reasoning = ReadabilityResponseParser.extract_content_between_tags(text, "reasoning")
        if not reasoning:
            logger.warning(f"Unable to parse <reasoning> tag. Raw response: {raw_response}")
            reasoning = ReadabilityResponseParser.DEFAULT_REASONING
        
        # Calculate overall score (average of dimensions)
        overall_score = sum(dimension_scores.values()) / len(dimension_scores)
        
        return {
            "dimension_scores": dimension_scores,
            "reasoning": reasoning,
            "overall_score": overall_score
        }

class ReadabilityPromptManager:
    """Handles creation of readability assessment prompts."""

    READABILITY_PROMPT_TEMPLATE = """
    You are an expert in evaluating the readability and complexity of texts. Your task is to assess the given text on several dimensions using a scale from 1 to 5, where 1 is the simplest and 5 is the most complex.

    When evaluating the text, you must:
    1. Assess each dimension independently using the defined 5-level scale.
    2. Provide a brief reasoning for your assessment.
    3. Ensure your evaluation is consistent and well-justified.

    The five dimensions and their levels (1 to 5) are defined as follows:

    **Vocabulary Complexity**:
    - Level 1: Very basic words, suitable for young children.
    - Level 2: Simple words, understandable by most adults.
    - Level 3: Moderate vocabulary, including some technical terms.
    - Level 4: Advanced vocabulary, with specialized terms.
    - Level 5: Highly technical or specialized vocabulary, requiring expert knowledge.
        
    **Syntactic Complexity**:
    - Level 1: Very simple sentence structures, short sentences.
    - Level 2: Basic sentence structures, mostly simple and compound sentences.
    - Level 3: Moderate complexity with a mix of simple and complex sentences.
    - Level 4: Complex sentence structures, with subordinate clauses and intricate syntax.
    - Level 5: Highly complex syntax, with nested clauses and sophisticated constructions.
        
    **Conceptual Density**
    - Level 1: Single, straightforward ideas presented one at a time.
    - Level 2: Few related concepts introduced at a manageable pace.
    - Level 3: Multiple concepts with clear connections between them.
    - Level 4: Many interrelated concepts requiring careful attention to follow.
    - Level 5: Dense with numerous abstract and interrelated concepts.

    **Background Knowledge**:
    - Level 1: No special knowledge needed beyond everyday experience.
    - Level 2: Basic familiarity with the subject area.
    - Level 3: General education in the domain or field discussed.
    - Level 4: Considerable domain knowledge required.
    - Level 5: Expert-level knowledge in the field necessary.
        
    **Cognitive Load**:
    - Level 1: Minimal effort to process and understand.
    - Level 2: Some attention needed but generally easy.
    - Level 3: Requires focus and moderate effort.
    - Level 4: Demands concentration and significant mental effort.
    - Level 5: Requires sustained intense concentration and analytical thinking.

    Below are examples to guide your assessment:

    **Example 1**  
    Text: 'The cat sat on the mat. It was a sunny day. The cat looked happy as it watched birds flying in the clear blue sky.'
    <root>
    <vocabulary_complexity>1</vocabulary_complexity>
    <syntactic_complexity>2</syntactic_complexity>
    <conceptual_density>1</conceptual_density>
    <background_knowledge>1</background_knowledge>
    <cognitive_load>1</cognitive_load>
    <reasoning>Vocabulary consists entirely of simple everyday words (level 1). Sentence structure includes both very basic sentences and one slightly more complex sentence with a dependent clause (level 2). Conceptually simple with only straightforward ideas about a cat and weather (level 1). Requires no special background knowledge (level 1). Overall cognitive demand remains minimal despite the slightly more complex sentence (level 1).</reasoning>
    </root>

    **Example 2**  
    Text: 'The rapid proliferation of digital technology has fundamentally transformed human interaction patterns. People now communicate across diverse platforms, sharing experiences instantaneously within global networks, which has altered our traditional concepts of community and privacy.'
    <root>
    <vocabulary_complexity>4</vocabulary_complexity>
    <syntactic_complexity>3</syntactic_complexity>
    <conceptual_density>4</conceptual_density>
    <background_knowledge>2</background_knowledge>
    <cognitive_load>3</cognitive_load>
    <reasoning>Vocabulary includes multiple advanced terms like "proliferation," "fundamentally transformed," and "instantaneously" (level 4). Sentence structures are moderately complex with compound elements but not highly intricate (level 3). Contains several dense, abstract concepts about technology's social impact packed into a short passage (level 4). Requires only basic familiarity with modern technology rather than specialized knowledge (level 2). Cognitive load is moderate due to the balance of complex vocabulary and concepts with familiar subject matter (level 3).</reasoning>
    </root>

    **Example 3**  
    Text: 'The quantum chromodynamics framework elucidates how quarks interact through gluons, manifesting the strong nuclear force via color charge confinement and asymptotic freedom properties. This mathematical approach successfully reconciles quantum field theory with experimental observations of hadron behavior.'
    <root>
    <vocabulary_complexity>5</vocabulary_complexity>
    <syntactic_complexity>4</syntactic_complexity>
    <conceptual_density>5</conceptual_density>
    <background_knowledge>5</background_knowledge>
    <cognitive_load>5</cognitive_load>
    <reasoning>Vocabulary consists almost entirely of highly specialized physics terminology like "quantum chromodynamics," "gluons," and "asymptotic freedom" (level 5). Sentence structure is complex with multiple embedded clauses, though not reaching the most intricate possible level (level 4). Conceptual density is extremely high with numerous abstract physics concepts tightly packed together (level 5). Requires expert-level knowledge in theoretical physics to understand the concepts (level 5). Cognitive load is at the highest level due to the combination of specialized vocabulary, abstract concepts, and the mental effort needed to process the relationships between them (level 5).</reasoning>
    </root>

    Now, evaluate the following text:  

    ---  
    {text}  
    ---  

    Place your response between <root> and </root> tags in exactly this format:
    <root>  
    <vocabulary_complexity>score</vocabulary_complexity>  
    <syntactic_complexity>score</syntactic_complexity>  
    <conceptual_density>score</conceptual_density>  
    <background_knowledge>score</background_knowledge>  
    <cognitive_load>score</cognitive_load>  
    <reasoning>brief explanation</reasoning>  
    </root>  

    Only include scores as integers between 1 and 5 within the tags.
    Ensure each tag is properly closed with the corresponding closing tag.  
    Do not include any additional text outside the <root> and </root> tags. Use only the specified XML format.
    """

    @staticmethod
    def create_prompt(text: str) -> str:
        """Create a prompt for readability evaluation."""
        return ReadabilityPromptManager.READABILITY_PROMPT_TEMPLATE.format(text=text)


class LLMReadabilityClassifier:
    """Evaluates text readability using LLM with XML-based formatting."""
    
    def __init__(self, model_path: str):
        """Initialize the LLMReadabilityClassifier with a TurboMind model."""
        engine_config = TurbomindEngineConfig(
            model_format='awq',
            cache_max_entry_count=0.75,
            session_len=4092,
            tp=1,
            enable_prefix_caching=True
        )
        self.pipe = pipeline(
            model_path, 
            backend_config=engine_config, 
            chat_template_config=None
        )
        # Optimized generation config
        self.gen_config = GenerationConfig(
            max_new_tokens=2048,
            top_k=1,
            temperature=0.0,
        )
        
    def predict_batch(self, texts: List[str], batch_size: int = 1000) -> List[Dict[str, Any]]:
        """Process batches with a progress bar using tqdm."""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting with LLM"):
            batch_texts = texts[i:i + batch_size]
            prompts = [ReadabilityPromptManager.create_prompt(t) for t in batch_texts]
            
            # Process the batch
            responses = self.pipe(prompts, gen_config=self.gen_config)
            
            # Parse results
            batch_results = [ReadabilityResponseParser.parse_readability_response(r.text.strip()) 
                            for r in responses]
            results.extend(batch_results)
            
        return results

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Evaluate readability for a single text."""
        return self.predict_batch([text])[0]

    def __del__(self):
        """Clean up pipeline resources."""
        del self.pipe


def main():
    model_path = "/beegfs/client/default/dl-models/turbomind/deepseek-r1-distill-llama-70b-awq-TurboMind"
    classifier = LLMReadabilityClassifier(model_path)
    
    # Example texts
    texts = [
        "Through September 2013, we did computer searches for studies of programs to improve condom use. We wrote to researchers for missing data. The studies could have various designs. The education program addressed preventing pregnancy and HIV/STI. The intervention was compared with a different program, usual care, or no intervention. The studies had a clinical outcome such as pregnancy, HIV, or STI tests. We did not use self-reports of condom use. We found seven randomized trials. Six assigned groups (clusters) and one randomized individuals. Four trials took place in African countries, two in the USA, and one in England. The studies were based in schools, community settings, a clinic, and a military training setting. Five trials examined pregnancy, four studied HIV and HSV-2 (herpes), and three assessed other STI. We found no major differences between study groups for pregnancy or HIV. Some results were seen for STI outcomes. Two studies showed fewer HSV-2 cases with the behavioral program compared to the control group. One also reported fewer cases of syphilis and gonorrhea with the behavioral program plus STI management. Another study reported a higher gonorrhea rate for the intervention group. The researchers believed the result was due to a subgroup that did not have the full program. We found little clinical effect of improving condom use. The studies provided moderate to low quality information. Losses to follow up were high. We need good programs on condom use to prevent pregnancy and HIV/STI. Programs should be useful for settings with few resources. Interventions should be tested with valid outcome measures."
    ]
    
    logger.info("Starting predictions...")
    
    # Run evaluation
    start_time = time.time()
    results = classifier.predict_batch(texts)
    elapsed = time.time() - start_time
    
    # Report results
    logger.info(f"Total time for {len(texts)} samples: {elapsed:.2f} seconds")
    logger.info(f"Average time per sample: {elapsed/len(texts):.2f} seconds")
    
    import json
    logger.info("First sample result:")
    logger.info(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    main()