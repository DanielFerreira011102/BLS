---
title: "Plug and Play Language Models: A Simple Approach to Controlled Text Generation"
authors: Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, Rosanne Liu
year: 2020
database: arXiv
citekey: dathathri2020plugplaylanguagemodels
tags:
  - ctg
  - plug-and-play
  - bow
  - latent-space-manipulation
  - discriminators
  - pplm
url: https://arxiv.org/abs/1912.02164
file: "[[Plug and Play Language Models - A Simple Approach to Controlled Text Generation.pdf]]"
---

>[!title]
Plug and Play Language Models: A Simple Approach to Controlled Text Generation

>[!year]
2020

>[!author]
Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, Rosanne Liu


------------------------------------

### Summary

- The paper introduces PPLM (Plug and Play Language Models), a simple approach that combines pre-trained language models with attribute controllers for controlled text generation.

- Rather than retraining or fine-tuning large language models, PPLM uses small attribute models (discriminators or bag-of-words) to guide text generation.

- The method updates latent representations of the language model through gradients from the attribute models while maintaining fluency.

- PPLM demonstrates control over various attributes including topics, sentiment, and detoxification without compromising text quality.

- The approach allows flexible combination of multiple attribute controllers during inference.

- The method achieves comparable or better results than baseline approaches that require full model retraining.

------------------------------------

### Research question

How can we achieve controlled text generation with pre-trained language models without requiring expensive model retraining or architectural modifications, while maintaining text fluency and allowing for flexible attribute control?

------------------------------------

### Context

The paper addresses a significant challenge in natural language generation - controlling attributes of generated text. While large transformer-based language models have shown impressive generation capabilities, controlling specific attributes (like topic or sentiment) typically requires either architectural modifications or fine-tuning on attribute-specific data. This is expensive and inflexible. The paper's approach is particularly relevant as it offers a simple, plug-and-play solution that works with any transformer-based text generator and can be combined with any differentiable attribute model. This has important implications for making language models more controllable and useful in practical applications.

------------------------------------

### Methodology

The PPLM methodology introduces a novel approach to controlled text generation that operates through careful manipulation of a language model's latent space. At its core, the method combines a pre-trained language model with attribute models without requiring any modification to the base model's parameters.

#### Latent Space Manipulation

The primary mechanism involves updating the history matrix Ht, which contains key-value pairs from previous generations. Rather than modifying the entire model architecture or retraining, PPLM performs updates to this historical context through gradient-based optimization. These updates effectively recontextualize the past, guiding future token generation toward desired attributes while maintaining coherence with the language model's learned patterns.

#### Attribute Controllers

The framework supports two primary types of attribute controllers:

1. Bag of Words (BoW) Controllers: These represent perhaps the simplest form of control, yet prove remarkably effective. A BoW controller computes the log likelihood of generating any word from a predefined set related to the desired attribute. Despite their simplicity, these controllers can effectively steer generation toward desired topics without any training.

2. Discriminator Controllers: These are more sophisticated controllers consisting of small neural networks (typically single-layer) trained to recognize specific attributes. The discriminator operates on the mean of embeddings across time. These controllers require training but can capture more nuanced attributes like sentiment or style.

#### Maintaining Fluency

A critical challenge in controlled generation is maintaining text fluency while achieving desired attributes. PPLM employs multiple mechanisms to address this:

1. KL Divergence Minimization: The method includes a KL divergence term between modified and unmodified language model distributions,.

2. Post-norm Geometric Mean Fusion: The final sampling distribution combines the controlled and uncontrolled distributions.

3. Adaptive Gradient Normalization: For BoW controllers, gradient updates are normalized by the maximum gradient norm over time, reducing update magnitude when attribute words are less likely to appear.

#### Implementation Details

The method implements a finite horizon update window (typically 5 tokens) to maintain computational efficiency and prevent long-range degeneration. Early stopping of latent updates can be employed to prevent repetitive text when control strength is high.

------------------------------------

### Findings

- The PPLM approach successfully controlled text generation across multiple attributes (topics, sentiment, detoxification) while maintaining fluency, as demonstrated through both automated and human evaluation metrics.

- Using gradient-based latent updates proved more effective than simple reranking approaches, showing the importance of actually steering the model's internal representations rather than just filtering outputs.

- The method achieved comparable or better performance than baseline approaches like CTRL and GPT2-FT-RL, despite not requiring any model retraining and using much smaller attribute models.

- Multiple attribute controllers could be successfully combined during generation, allowing for complex control over generated text attributes.

- The approach effectively reduced toxicity in language model outputs, both for natural prompts and adversarial triggers, demonstrating its utility for practical applications.

------------------------------------

### Discussion

The PPLM framework presents several compelling opportunities for developing controllable medical question-answering systems, while also raising important considerations and challenges.

#### Complexity Control Architecture

For medical QA applications, the PPLM framework could be adapted to control explanation complexity through a multi-layered approach. A primary discriminator could be trained on medical texts labeled with complexity levels, while multiple BoW controllers could handle different aspects of medical terminology. The ability to combine controllers suggests the possibility of separate control over technical vocabulary, conceptual depth, and explanation structure.

The framework's gradient-based latent manipulation is particularly appealing for medical applications as it allows for:

1. Continuous complexity adjustment during generation.

2. Preservation of the base model's medical knowledge.

3. Integration of multiple control aspects without retraining.

#### Implementation Considerations for Medical QA

The implementation of PPLM for medical QA would require careful attention to several aspects. First, the attribute models would need to be designed specifically for medical complexity control. This might involve:

- Training discriminators on carefully curated medical texts spanning different complexity levels.

- Developing hierarchical BoW controllers for medical terminology.

- Creating specialized controllers for maintaining medical accuracy.

The method's hyperparameter sensitivity becomes particularly crucial in medical applications. The balance between control strength, KL divergence weight, and geometric mean fusion would need careful tuning to ensure generated explanations remain both accurate and appropriately calibrated to the desired complexity level.

#### Technical Challenges and Opportunities

The current PPLM implementation's token-by-token update mechanism, while effective, may present computational challenges for real-time medical QA systems. However, this also offers opportunities for innovative optimizations, such as:

- Caching commonly used latent updates for standard medical concepts.

- Developing more efficient update schemes for medical terminology.

- Implementing adaptive control strength based on content complexity.

The framework's flexibility in combining controllers suggests the possibility of developing specialized medical controllers for different aspects of explanation complexity. For instance, separate controllers could handle:

- Technical terminology adjustment.

- Conceptual depth modulation.

- Explanation structure adaptation.

- Citation and evidence-level control.

#### Future Directions and Potential Enhancements

For medical QA applications, several enhancements to the basic PPLM framework might be beneficial:

1. Development of hierarchical control mechanisms that can maintain consistency across longer medical explanations.

2. Integration of medical knowledge graphs to ensure accuracy during complexity adjustment.

3. Implementation of feedback mechanisms to dynamically adjust control strength based on user comprehension signals.

The framework's ability to maintain the base model's fluency while adding control suggests it could be particularly valuable for medical applications where maintaining clarity and accuracy across different complexity levels is crucial. However, extensive validation would be needed to ensure that complexity adjustments don't introduce medical inaccuracies or potentially harmful information.

------------------------------------

### Remarks & Limitations

- The approach requires careful tuning of hyperparameters to balance attribute control and fluency, which might need special consideration for medical content where accuracy is crucial.

- While the method works well for controlling high-level attributes, it might need adaptation for controlling fine-grained aspects of medical explanations, such as maintaining technical accuracy while simplifying language.

- The current implementation updates latent representations at each generation step, which could impact generation speed in real-time applications - this might need optimization for interactive medical QA systems.

- The approach occasionally shows degeneration (repetitive text) when attribute control is too strong, suggesting a need for careful calibration in medical applications where clarity is essential.

- While the method allows for multiple attribute controllers, there might be challenges in ensuring these controllers work harmoniously for medical content where different aspects of complexity (terminology, concept depth, explanation structure) need to be balanced.

------------------------------------

### Citation

```
@misc{dathathri2020plugplaylanguagemodels,
      title={Plug and Play Language Models: A Simple Approach to Controlled Text Generation}, 
      author={Sumanth Dathathri and Andrea Madotto and Janice Lan and Jane Hung and Eric Frank and Piero Molino and Jason Yosinski and Rosanne Liu},
      year={2020},
      eprint={1912.02164},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1912.02164}, 
}
```