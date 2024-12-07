---
title: "FAST: Improving Controllability for Text Generation with Feedback Aware Self-Training"
authors: Junyi Chai, Reid Pryzant, Victor Ye Dong, Konstantin Golobokov, Chenguang Zhu, Yi Liu
year: 2022
database: arXiv
citekey: chai2022fastimprovingcontrollabilitytext
tags:
  - ctg
  - self-training
  - inverse-propensity-score
  - ips
  - spurious-correlations
  - fast
  - data-resampling
  - counterfactual-data-augmentation
  - validation-filtering
  - ctrl
  - rouge
  - bart
  - roberta
  - transformers
  - controlled-text-generation
url: https://arxiv.org/abs/2210.03167
file: "[[FAST - Improving Controllability for Text Generation with Feedback Aware Self-Training.pdf]]"
---

>[!title]
FAST: Improving Controllability for Text Generation with Feedback Aware Self-Training

>[!year]
2022

>[!author]
Junyi Chai, Reid Pryzant, Victor Ye Dong, Konstantin Golobokov, Chenguang Zhu, Yi Liu


------------------------------------

### Summary

- The paper addresses a critical flaw in controllable text generation where models fail to properly respect control codes due to spurious correlations in training data.

- It introduces two techniques: IPS (Inverse Propensity Score) resampling and FAST (Feedback Aware Self-Training).

- FAST generates counterfactual versions of training examples and uses feedback to filter poor generations.

- The methods are evaluated on three tasks: news headline generation, meta-review generation, and search ad generation.

- Results show FAST significantly improves controllability while maintaining or improving text quality.

- The approach outperforms existing methods like PPLM and GeDi.

------------------------------------

### Research question

How can we address and mitigate the issue of spurious correlations in controllable text generation systems where models incorrectly rely on input context rather than control codes for attribute selection, and how can we improve both controllability and generation quality?

------------------------------------

### Context

The paper addresses a critical issue in controllable text generation systems that use control codes to direct output properties such as style, length, or tone. While these systems are widely used, they often fail to properly respect the control codes due to unwanted correlations in training data between the input context and output attributes. This undermines the effectiveness of controllable generation systems across various applications. The work is particularly relevant as controllable generation becomes increasingly important in practical applications, from content adaptation to personalized text generation.

------------------------------------

### Methodology

The methodology can be broken down into several key components:

#### Problem Identification

The researchers first establish the existence of spurious correlations in controllable text generation by demonstrating how models learn to predict output attributes from input context, independent of control codes. They use statistical analysis and classification experiments to quantify these unwanted correlations.

#### IPS Resampling Approach 

The first proposed solution involves resampling the training data using inverse propensity scores. This technique gives more weight to rare context-attribute combinations, effectively balancing the dataset to reduce spurious correlations. The propensity scores are estimated using a RoBERTa-based classifier trained to predict attributes from context.

#### FAST Algorithm

The second and more sophisticated solution involves a four-step process:

1. Training an initial controlled generation model.

2. Using this model to generate counterfactual versions of each example using different control codes.
 
3. Applying a feedback mechanism to filter out generations that don't match the intended attributes.

4. Retraining the model on the combined original and filtered counterfactual data.

#### Evaluation Framework

The methodology includes a comprehensive evaluation approach using both automated metrics (ROUGE scores, attribute accuracy) and human evaluation. The researchers test their methods on three diverse tasks to demonstrate generalization.

------------------------------------

### Findings

- FAST consistently outperforms existing methods across all three tasks, showing significant improvements in both controllability and language quality, particularly on the meta-review and advertising tasks.

- The IPS resampling method, while effective at reducing spurious correlations, can sometimes lead to decreased language quality due to the duplication of training examples.

- The feedback component in FAST proves crucial, as removing it leads to significant drops in control accuracy (9%, 7%, and 12% absolute differences across the three tasks).

- The effectiveness of FAST remains robust even when using weaker classifiers for feedback, suggesting the method's practical applicability in real-world scenarios.

- Human evaluation confirms the automated metrics, showing FAST improves both language quality and controllability in generated content.

------------------------------------

### Discussion

The foundation of this system lies in two critical validators: the **complexity classifier** and the **medical accuracy validator**. Let's start with the complexity classifier. To build this, we need to create a comprehensive corpus of **medical texts representing different complexity levels**. This corpus should include everything from basic patient education materials to specialized medical research papers, creating a natural spectrum of complexity. Medical textbooks from different educational levels provide an excellent middle ground, while clinical guidelines and research papers represent the higher end of complexity.

The **complexity classifier** needs to understand both **linguistic** and **medical** aspects of complexity. On the linguistic side, it analyzes sentence structure, vocabulary sophistication, and overall readability. On the medical side, it examines the density of technical terminology, the complexity of medical concepts discussed, and the use of scientific notation or statistical terminology. By training on this diverse corpus, the classifier learns to assign **continuous complexity scores between 0 and 1**, where 0 represents the most accessible, patient-friendly language and 1 represents highly technical, specialist-level content.

The **medical accuracy validator** serves as the guardian of factual correctness. It needs to be built upon a robust **medical knowledge base** that contains not just facts, but also relationships between medical concepts, common misconceptions, and potential errors. When evaluating generated text, this validator extracts key medical claims and compares them against its knowledge base, checking not just for factual accuracy but also for logical consistency and appropriate relationships between medical concepts. It's crucial that this validator can distinguish between **simplification** (which is acceptable) and **inaccuracy** (which isn't).

With these validators in place, we can implement **complexity control** in the generation process. One effective approach is **prefix-tuning**, where we add learnable prefixes to the model input that guide the generation process. These prefixes are trained to correspond to different complexity levels, allowing smooth interpolation between them. During generation, we can adjust these prefixes based on the desired complexity level, effectively steering the model's output while maintaining medical accuracy.

Another approach is to implement control during the **decoding process**. This involves scoring candidate tokens for both their relevance to the medical content and their contribution to the target complexity level. The generation process then balances these factors, ensuring that the output maintains accuracy while achieving the desired complexity level.

The **training process** begins with fine-tuning a medical language model on question-answer pairs. During training, we generate multiple versions of each answer at different complexity levels, using our validators to ensure both appropriate complexity and medical accuracy. Examples that fail either validation are filtered out, while successful generations are added to an augmented training dataset. This process continues iteratively, gradually improving the model's ability to generate accurate medical information at any desired complexity level.

The **evaluation system** needs to be equally sophisticated, combining **automated metrics** with **human expertise**. Automated evaluation measures complexity accuracy, medical correctness, and information preservation across complexity levels. Human evaluation, particularly from medical professionals and target users, provides crucial feedback on the practical usefulness and clarity of the generated responses.

The final system should allow users to **dynamically adjust** the complexity of responses through an intuitive interface, perhaps using a slider control. As users adjust the complexity level, the system should smoothly transition between different versions of the response, maintaining medical accuracy while adjusting the level of technical detail and linguistic complexity.

The greatest challenge in this system is maintaining the delicate balance between simplification and accuracy. When simplifying medical information, we must ensure we don't introduce inaccuracies or oversimplifications that might mislead. Conversely, when increasing complexity, we need to ensure we're adding meaningful technical detail rather than just making the language more complicated.

------------------------------------

### Remarks & Limitations

- The IPS resampling method faces limitations with large pretrained models, which are already relatively efficient at learning from small amounts of counterfactual examples.

- The approach assumes that linguistic attributes should be independent of context, which may not always be true in real-world applications, potentially limiting its applicability in certain scenarios.

- The FAST method may struggle when training and pretraining data are drastically different, as the quality of counterfactual generations could be poor and propagate errors.

- The evaluation metrics used might not fully capture the nuanced aspects of controlled generation, particularly in terms of semantic preservation across different control settings.

- The computational cost of generating counterfactual examples and applying feedback mechanisms could be prohibitive for large-scale applications.

- The approach relies on the availability of reliable classifiers for attribute detection, which might not always be available or might be difficult to develop for certain attributes or domains.

- The method's effectiveness might vary depending on the specific nature of the spurious correlations present in the training data, making it potentially less reliable for certain types of control tasks.

------------------------------------

### Citation

```
@misc{chai2022fastimprovingcontrollabilitytext,
      title={FAST: Improving Controllability for Text Generation with Feedback Aware Self-Training}, 
      author={Junyi Chai and Reid Pryzant and Victor Ye Dong and Konstantin Golobokov and Chenguang Zhu and Yi Liu},
      year={2022},
      eprint={2210.03167},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2210.03167}, 
}
```