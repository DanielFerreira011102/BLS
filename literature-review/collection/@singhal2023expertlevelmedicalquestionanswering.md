---
title: Towards Expert-Level Medical Question Answering with Large Language Models
authors: Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, Mike Schaekermann, Amy Wang, Mohamed Amin, Sami Lachgar, Philip Mansfield, Sushant Prakash, Bradley Green, Ewa Dominowska, Blaise Aguera y Arcas, Nenad Tomasev, Yun Liu, Renee Wong, Christopher Semturs, S. Sara Mahdavi, Joelle Barral, Dale Webster, Greg S. Corrado, Yossi Matias, Shekoofeh Azizi, Alan Karthikesalingam, Vivek Natarajan
year: 2023
database: arXiv
citekey: singhal2023expertlevelmedicalquestionanswering
tags:
  - Med-PaLM/2
  - Med-PaLM
  - PaLM
  - PaLM/2
  - MedQA-US
  - Ensemble_Refinement
  - MedMCQA
  - PubMedQA
  - MMLU-Med
  - HealthSearchQA
  - Instruction_Tuning
  - Adversarial_Testing
  - Alignment
  - Physician_Comparison
  - LiveQA
  - MedicationQA
  - Pairwise_Ranking
  - Human_Evaluation
  - Multi-choice
  - Open-ended
  - MultiMedQA
  - USMLE
url: https://arxiv.org/abs/2305.09617
file: "[[Towards Expert-Level Medical Question Answering with Large Language Models.pdf]]"
---

>[!title]
Towards Expert-Level Medical Question Answering with Large Language Models

>[!year]
2023

>[!author]
Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, Mike Schaekermann, Amy Wang, Mohamed Amin, Sami Lachgar, Philip Mansfield, Sushant Prakash, Bradley Green, Ewa Dominowska, Blaise Aguera y Arcas, Nenad Tomasev, Yun Liu, Renee Wong, Christopher Semturs, S. Sara Mahdavi, Joelle Barral, Dale Webster, Greg S. Corrado, Yossi Matias, Shekoofeh Azizi, Alan Karthikesalingam, Vivek Natarajan


------------------------------------

### Summary

- The paper discusses the development of Med-PaLM 2, a large language model designed for answering medical questions at a level comparable to physicians.

- The first version, Med-PaLM, made significant progress in medical question answering, but there was still a gap in quality when compared to physicians' answers.

- Med-PaLM 2 uses an advanced model (PaLM 2) with improvements in medical reasoning, and applies _ensemble refinement_, a new strategy to refine answers by generating multiple reasoning paths and improving them.

- The model achieves state-of-the-art performance on several medical question-answering benchmarks, achieving an accuracy of **86.5%** on MedQA (a dataset of USMLE-style medical exam questions) and surpassing Med-PaLM by over **19%**.

- Physicians evaluated the model’s answers to real-world medical questions, finding that Med-PaLM 2’s responses were preferred over physicians’ answers in **eight out of nine clinical utility axes** (e.g., factuality and reasoning).

- Med-PaLM 2's answers were rated as reflecting medical consensus **72.9%** of the time compared to physician answers.

- Two new datasets of complex, adversarial medical questions were used to test the model’s safety and robustness, showing that Med-PaLM 2 significantly outperformed the previous version in areas like minimizing potential harm.

- Med-PaLM 2 demonstrated a low risk of harm in **90.6%** of its answers, compared to **79.4%** for Med-PaLM.

- The model still requires further validation in real-world settings and ongoing refinement to handle nuanced medical scenarios safely and reliably

------------------------------------

### Research question

How can large language models (LLMs) be improved to achieve physician-level performance in answering medical questions across various clinical and consumer health domains?

------------------------------------

### Context

The paper emerged from the broader context of applying artificial intelligence (AI) to healthcare, where AI systems have shown remarkable progress in complex tasks such as protein folding and games like Go. In medicine, answering clinical questions at the level of trained physicians has long been seen as a "grand challenge."

The relevance of the work lies in the growing potential of large language models (LLMs) to transform healthcare by retrieving medical knowledge, reasoning through it, and providing accurate, reliable answers to medical questions. The importance of this paper stems from the fact that existing models, while capable of passing medical exams like the USMLE, still fall short in matching the depth, safety, and alignment of human-generated answers, especially in critical and nuanced real-world settings.

The goal of this research is to close the performance gap between AI models and physicians in medical question answering. By improving the model's reasoning capabilities, factual accuracy, and alignment with clinical standards, the authors aim to create a system that can assist doctors and improve access to high-quality medical information, benefiting patients and healthcare providers alike​(Towards Expert-Level Me…).

------------------------------------

### Methodology

The paper develops **Med-PaLM 2**, a large language model (LLM) designed to answer medical questions at a near-physician level of accuracy. Med-PaLM 2 improves upon previous models by integrating domain-specific fine-tuning and introducing a novel **ensemble refinement** strategy for better medical reasoning. The methodology focuses on multiple-choice benchmarks and long-form medical question-answering across clinical and consumer health topics.

![[5jBpZ5zw16atNxWzxbETEbi4YAueyjvo.png|center]]

#### Datasets

Med-PaLM 2 is evaluated using several datasets that test its ability to answer medical questions, both in multiple-choice and open-ended formats:

1. **Multiple-choice Datasets**:

    - **MedQA (USMLE)**: 1,273 questions modeled after the U.S. Medical Licensing Exam, designed to assess general medical knowledge.
    
    - **MedMCQA**: 4,183 multiple-choice questions used in Indian medical entrance exams.
    
    - **PubMedQA**: 500 closed-domain questions with yes/no/maybe answers derived from PubMed abstracts.
    
    - **MMLU Clinical Topics**: A variety of clinical knowledge datasets, including subsets focused on anatomy, genetics, and professional medicine.
    
1. **Long-form Question Datasets**:

    - **MultiMedQA 140**: A dataset of 140 long-form questions sampled from HealthSearchQA, LiveQA, and MedicationQA, addressing consumer health concerns.
    
    - **MultiMedQA 1066**: An expanded version with 1,066 consumer health questions.
    
1. **Adversarial Datasets**:

    - **General Adversarial Dataset**: A set of 58 questions designed to challenge the model with complex issues like drug use, mental health, and medical misinformation.
    
    - **Health Equity Adversarial Dataset**: A set of 182 questions focused on health disparities and bias, covering topics like access to healthcare and structural inequalities.

#### Evaluation Metrics

To measure the performance of Med-PaLM 2, the paper uses the following metrics:

1. **Accuracy**: The primary metric for evaluating performance on multiple-choice datasets, indicating the proportion of correct answers.

2. **Human Evaluation**: Long-form answers are assessed by physicians and laypeople across several axes, including:

    - **Factuality**: Whether the answer aligns with medical consensus.
    
    - **Comprehension**: The model’s understanding of the question.
    
    - **Reasoning**: The quality of medical reasoning demonstrated in the answer.
    
    - **Risk of Harm**: The likelihood that an answer could lead to harm due to incorrect or misleading information.

#### Evaluation Settings

The model was evaluated under multiple settings to assess its performance across different types of medical questions:

1. **Multiple-choice evaluation**: The model was tested using few-shot, chain-of-thought, and ensemble refinement prompting techniques to select the correct answers from predefined options in multiple-choice tests.

2. **Long-form evaluation**: The model’s ability to generate detailed answers to open-ended questions was tested. Human raters—both physicians and laypeople—evaluated these responses based on clarity, relevance, and safety.

3. **Adversarial Performance**:  Med-PaLM 2 was also tested on two **adversarial datasets** designed to challenge the model with difficult or sensitive medical questions, particularly to detect bias or potential harm.
        
4. **Overlap Analysis**:  An **overlap analysis** was performed to check if Med-PaLM 2’s performance was influenced by memorization from its training data. 

    - **Defining Overlap**: Questions were flagged as overlapping if any part of the question (excluding answers) matched 512 contiguous characters from the model’s training corpus.
    
    - **Process**: The team searched for overlapping segments between evaluation datasets and the corpus used to pretrain the model.
    
    - **Purpose**: The goal was to ensure the model’s high performance wasn’t the result of memorization from training data but rather its generalization ability.
    
Results showed minimal differences between performance on overlapping and non-overlapping questions, confirming that Med-PaLM 2’s strong performance comes from generalization rather than memorization.

#### Preprocessing

The study doesn't explicitly mention preprocessing steps. However, the instruction finetuning process involves preparing the training data from MultiMedQA datasets with specific mixture ratios:

- MedQA: 37.5%

- MedMCQA: 37.5%

- LiveQA: 3.9%

- MedicationQA: 3.5%

- HealthSearchQA: 17.6%

These datasets were combined to train the model in handling a diverse range of medical questions.
### Models

Med-PaLM 2 was built on PaLM 2, an advanced large language model, and further trained on medical data to specialize in clinical reasoning. The model's workflow incorporates the following key techniques:
#### Ensemble Refinement

This novel prompting strategy involves:

1. Generating multiple initial answers (reasoning paths) to a given question.
2. Providing these initial answers back to the model as context.
3. Asking the model to generate a final, refined answer based on the collective insights from the initial answers.

This approach allows the model to consider multiple perspectives before producing a final response, potentially improving the quality and accuracy of its medical reasoning.

#### Chain-of-Thought and Self-Consistency

- Chain-of-Thought: This technique prompts the model to show its work by breaking down its reasoning into step-by-step explanations. For medical questions, this might involve listing relevant facts, considering differential diagnoses, and explaining the rationale for the final answer.
- Self-Consistency: This method involves:
    
    1. Generating multiple independent answers to the same question.
    2. Selecting the most common answer as the final response.
    

This approach helps mitigate inconsistencies and can improve overall accuracy, especially for complex medical queries.

#### Few-Shot Prompting

Few-shot prompting involves providing the model with a small number of example question-answer pairs before asking it to answer a new question. For Med-PaLM 2, this likely included:

1. Selecting relevant example medical questions and their correct answers.
2. Presenting these examples to the model along with the new question to be answered.
3. Prompting the model to answer the new question in a similar format to the examples.

This technique helps the model understand the expected format and style of answers, improving its performance across different types of medical questions.

### Training & Technology

The model was trained using NVIDIA GPUs, though specific details about the hardware configuration are not provided in the paper. The training process involved:

1. Starting with the pre-trained PaLM 2 model as a base.
2. Applying instruction finetuning using the MultiMedQA datasets with the specified mixture ratios.
3. Implementing the ensemble refinement, chain-of-thought, and few-shot prompting techniques during inference.

While the paper doesn't specify exact hyperparameters like learning rates or batch sizes, it's likely that these were carefully tuned to optimize performance on medical question-answering tasks. The extensive human evaluation process involved physicians and laypeople assessing the quality, safety, and clinical utility of the model's answers across various dimensions.

------------------------------------

### Findings

- **Med-PaLM 2** achieved an impressive accuracy of **86.5%** on USMLE-style questions from the MedQA dataset, marking a significant improvement of over 19% compared to the previous version, **Med-PaLM**. It also matched or exceeded state-of-the-art results on other datasets, including **MedMCQA**, **PubMedQA**, and **MMLU clinical topics**.
    
- **Long-form answers** from Med-PaLM 2 were generally rated **more favorably** than those generated by Med-PaLM or even physicians. Across **eight out of nine key evaluation axes**, Med-PaLM 2 performed better in areas such as **medical reasoning**, **factual accuracy**, and **safety**. Notably, **72.9%** of Med-PaLM 2’s answers were judged to better reflect **medical consensus** compared to physician answers.
    
- In evaluations using **adversarial medical question sets**, Med-PaLM 2 significantly outperformed Med-PaLM on all axes, particularly in reducing the **risk of harm**. Specifically, **90.6%** of Med-PaLM 2’s answers were rated as having a **low risk of harm**, compared to **79.4%** for Med-PaLM.
    
- Med-PaLM 2’s responses were rated better than physician-generated responses in reflecting **medical consensus**, demonstrating superior **reading comprehension**, and showing better **knowledge recall**.
    
- Med-PaLM 2 displayed substantial improvements in all quality dimensions for long-form question-answering, particularly in consumer health-related and **adversarial** questions. These improvements were also evident in questions focused on **health equity**, further demonstrating the model's ability to handle sensitive and complex topics.

------------------------------------

### Discussion

Med-PaLM 2 shows **strong performance**, either approaching or surpassing state-of-the-art results across multiple medical question-answering benchmarks, including **MedQA** and **MedMCQA**. This suggests that Med-PaLM 2 is capable of answering medical questions with **high accuracy**, **knowledge recall**, and **reasoning ability**.

The model's significant advancements in **long-form medical question-answering** are particularly noteworthy. Med-PaLM 2’s answers were consistently preferred over those generated by Med-PaLM and even **physicians**, demonstrating its enhanced ability to understand and reason through **complex medical issues**. These improvements were especially pronounced in handling **challenging adversarial question sets**, highlighting the model's capacity to deal with difficult and nuanced cases.

The fact that Med-PaLM 2’s answers were preferred over physician responses on multiple axes—such as **factuality** and **alignment with medical consensus**—suggests that it holds promise for **real-world applications**, particularly in the realm of **consumer medical question-answering**. However, the authors emphasize that there is still a need for ongoing development, especially as large language models (LLMs) become more proficient in structured knowledge tests.

Despite its strong performance, the paper underlines the importance of continued evaluation and **refinement** of these models, especially in mitigating **factual inaccuracies** or the inclusion of **irrelevant information**. The focus on creating robust, safe long-form answers suggests that models like Med-PaLM 2 will still require **ongoing attention** to ensure they remain **safe** and **reliable** in clinical settings.

In conclusion, the paper positions Med-PaLM 2 as an **advanced model** with the potential to make significant contributions to **medical question-answering**, particularly in **consumer health**. However, continued evaluation, particularly for **safety** and **accuracy**, will be essential as these models are integrated into **real-world clinical environments**.

------------------------------------

### Remarks & Limitations

- **Measuring Empathy**: The study recognizes that evaluating **empathy** in the model's answers is important. However, the current methods don't fully capture how well the model conveys empathy, which may be crucial in healthcare settings.
    
- **Non-validated Rubric**: The rating rubric used for evaluating the model's performance is not a **formally validated** instrument. Although the **inter-rater reliability** (agreement between evaluators) was high, more research is needed to develop standardized tools for evaluating **Large Language Models (LLMs)** in medical tasks.
    
- **Limited Physician Evaluations**: The physicians providing answers weren't given specific **clinical scenarios** or tailored instructions regarding the communication style for their audience (e.g., laypeople). This could limit how applicable the evaluation results are to **real-world clinical settings**.
    
- **Answer Length Discrepancies**: Med-PaLM 2's answers were generally longer than those provided by physicians, which might have affected the evaluation favorably. The extra length may have contributed to higher ratings but doesn't necessarily indicate superior quality.
    
- **Inter-Physician Variability**: Only one answer per physician was evaluated for each question. In reality, physicians may give **different responses** to the same question, so the evaluation might not reflect the full variability among **human-generated answers**.
    
- **Scope of Adversarial Testing**: The adversarial datasets used to assess the model’s performance on **sensitive topics** (like health equity) are relatively limited in scope. A broader, more **systematic evaluation** of health equity concerns would be needed for a comprehensive assessment.
    
- **Evaluation Focus**: The study highlights the need for further research into comparing LLM-generated answers with a broader range of **physician-generated responses**, including their medical background and lived experience. This could provide a more nuanced understanding of how LLMs compare to human experts.

------------------------------------

### Citation

```
@misc{singhal2023expertlevelmedicalquestionanswering,
      title={Towards Expert-Level Medical Question Answering with Large Language Models}, 
      author={Karan Singhal and Tao Tu and Juraj Gottweis and Rory Sayres and Ellery Wulczyn and Le Hou and Kevin Clark and Stephen Pfohl and Heather Cole-Lewis and Darlene Neal and Mike Schaekermann and Amy Wang and Mohamed Amin and Sami Lachgar and Philip Mansfield and Sushant Prakash and Bradley Green and Ewa Dominowska and Blaise Aguera y Arcas and Nenad Tomasev and Yun Liu and Renee Wong and Christopher Semturs and S. Sara Mahdavi and Joelle Barral and Dale Webster and Greg S. Corrado and Yossi Matias and Shekoofeh Azizi and Alan Karthikesalingam and Vivek Natarajan},
      year={2023},
      eprint={2305.09617},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.09617}, 
}
```