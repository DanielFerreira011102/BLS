---
title: Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine
authors: Xiaorui Su, Yibo Wang, Shanghua Gao, Xiaolong Liu, Valentina Giunchiglia, Djork-Arné Clevert, Marinka Zitnik
year: 2024
database: arXiv
citekey: su2024knowledgegraphbasedagent
tags:
  - KGAREVION
  - KG
  - MedDDx
  - PubMedQA
  - BioASQ
  - MedQA-US
  - MMLU-Med
  - Llama/3/8B
  - Llama/3/1/8B
  - Fine-tuning
  - Knowledge_Completion
  - QA
  - Multi-choice
  - Open-ended
  - USMLE
  - CoT
  - Llama/2/7B
  - Llama/2/13B
url: https://arxiv.org/abs/2410.04660
file: "[[research-papers/Knowledge Graph Based Agent For Complex, Knowledge-intensive QA In Medicine.pdf|Knowledge Graph Based Agent For Complex, Knowledge-intensive QA In Medicine]]"
---

>[!title]
Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine


>[!year]
2024

>[!author]
Xiaorui Su, Yibo Wang, Shanghua Gao, Xiaolong Liu, Valentina Giunchiglia, Djork-Arné Clevert, Marinka Zitnik


------------------------------------

### Summary

- The paper introduces **KGAREVION**, an AI agent designed to handle complex, knowledge-intensive medical queries by using a combination of Large Language Models (LLMs) and **knowledge graphs (KGs)**.

- **KGAREVION** creates "triplets" from medical knowledge to ensure accuracy and relevance by grounding them in a medical knowledge graph.

- It improves on standard LLMs by offering more accurate answers to medical queries, addressing limitations like incorrect retrievals and misalignment with current scientific knowledge.

- Evaluated on multiple medical QA datasets, it outperforms baseline models with a **10.4% improvement** in accuracy on curated datasets.

- The model uses **multiple reasoning strategies** (rule-based, prototype-based, case-based), crucial for medical reasoning, which often involves analogy and context-specific knowledge.

- **Key contributions** include dynamic reasoning adaptation, structured knowledge verification through KGs, and the introduction of new medical QA datasets with increasing semantic complexity.


------------------------------------

### Research question

How can a knowledge graph-based AI agent be designed to improve the accuracy and reasoning of large language models when addressing complex, knowledge-intensive medical questions by integrating both structured and unstructured medical knowledge?

------------------------------------

### Context

![[UAlDyxI6ShVvjucTUmZh5yElTjTHV9Z4.png|center]]

The complexity of biomedical knowledge presents unique challenges for artificial intelligence (AI) systems, especially in the field of medicine. Unlike other scientific disciplines, such as physics or chemistry, **medical reasoning often involves analogy-based thinking and relies heavily on nuanced, context-specific information**. Medical professionals must draw on various reasoning strategies to diagnose diseases, prescribe treatments, and understand the mechanisms behind illnesses. For instance, models in medical research frequently involve using analogies between organisms to understand disease mechanisms that apply to humans.

This creates a need for AI systems that can handle **complex, knowledge-intensive queries** in medicine, where simple rule-based or fact-retrieval approaches are insufficient. Large language models (LLMs) like GPT-4 have demonstrated remarkable success in general-purpose tasks but struggle with the **highly specialized knowledge** required in the medical domain. These models may retrieve incorrect or incomplete information and fail to account for the specific medical context, such as patient demographics, disease-specific characteristics, or localized medical conditions.

Furthermore, existing models face challenges in integrating multiple types of evidence, including structured scientific knowledge from research studies and **tacit expert knowledge** based on real-world medical practice. This gap highlights the need for a more robust approach, combining the **structured knowledge** of medical concepts with the **flexibility** and **reasoning capabilities** of LLMs.

**Knowledge graphs (KGs)**, which represent complex relationships between medical concepts, have emerged as a valuable resource in AI for healthcare. However, existing systems that rely solely on KGs often lack the depth and adaptability to tackle complex medical reasoning, particularly when the knowledge in these graphs is incomplete or the reasoning requires integrating multiple medical concepts.

To address these challenges, the paper introduces **KGAREVION**, a new AI agent that uses both LLMs and KGs to answer complex medical questions accurately and flexibly. This agent was developed to improve upon previous models by grounding LLM-generated information in structured medical knowledge, ensuring accuracy and relevance in contexts where medical reasoning is especially difficult. This work came at a time when AI in healthcare was advancing rapidly, but still grappling with integrating complex medical knowledge into AI-driven solutions for real-world applications.

------------------------------------

### Methodology

The paper proposes **KGAREVION**, an AI agent designed to tackle complex, knowledge-intensive medical questions. KGAREVION works by combining the knowledge retrieval power of **Large Language Models (LLMs)** with the structured information of **Knowledge Graphs (KGs)**. The model has four main stages: Generate, Review, Revise, and Answer, each contributing to the identification of the correct medical answers.

![[ZIFhJVN3eM5EvHbY0YtOOJnzT0aCXF7R.png|center]]

#### Datasets

The paper evaluates the performance of the KGAREVION model using several datasets designed for medical question answering (QA). These datasets include:

1. **Gold-Standard Medical QA Datasets**:
    
    - **MMLU-Med**: Contains 1,089 multiple-choice questions covering various medical topics.
    - **MedQA-US**: Comprises 1,273 multiple-choice questions focused on U.S. medical licensing examinations.
    - **PubMedQA**: A dataset with 500 questions formatted as yes/no/maybe queries, derived from biomedical literature.
    - **BioASQ**: Includes 618 yes/no questions related to biological information.
    
1. **Newly Introduced Datasets**:

    - **MedDDx**: A new benchmark focused on differential diagnosis questions. It includes three difficulty levels:
        - **MedDDx-Basic**: 483 questions aimed at basic knowledge.
        - **MedDDx-Intermediate**: 1,041 questions requiring intermediate-level understanding.
        - **MedDDx-Expert**: 245 questions designed for advanced knowledge in medical reasoning.

The datasets were created to include a variety of question types and complexities, ensuring that the model is evaluated comprehensively across different medical scenarios.
#### Evaluation Metrics

The performance of KGAREVION is assessed using several key metrics:

1. **Accuracy**: The primary metric used to evaluate the model's performance, indicating the proportion of correct answers among all responses given by the model.
    
2. **Standard Deviation (std)**: This metric is reported alongside accuracy to provide insights into the variability of the model's performance across multiple evaluation runs.

#### Evaluation Settings

The evaluation was carried out in two distinct settings:

1. **Multi-choice reasoning**:
    - The model was tasked with selecting the correct answer from a list of candidate answers.
2. **Open-ended reasoning**:
    - In this setting, the model was required to generate an answer without being provided with predefined options. This setting closely mimics real-world medical scenarios where no answer options are given.

To further test the model’s capabilities, two additional scenarios were introduced:

- **Query Complexity Scenario (QSS)**: Evaluates how well the model handles questions involving a greater number of medical concepts.
- **Semantic Complexity Scenario (CSS)**: Tests the model’s ability to discern between closely related or semantically similar medical concepts.

By considering these factors, KGAREVION is shown to maintain accuracy and robustness, even when the complexity of medical queries increases.
#### Model

KGAREVION integrates LLM-generated knowledge with a grounded knowledge graph to handle medical reasoning. The model's workflow consists of four key actions:

1. **Generate Action**:
    
    - The LLM generates relevant knowledge triplets based on the input question.
    - For **choice-aware questions** (with predefined answers), triplets are generated for each candidate answer based on the extracted medical concepts.
    - For **non-choice-aware questions** (yes/no/maybe type), triplets are generated solely based on the question's medical concepts.
2. **Review Action**:
    
    - This action checks the correctness of the generated triplets by aligning them with the knowledge graph. The LLM is fine-tuned on a **knowledge completion task**, which involves understanding the structure and relationships between medical entities within the KG.
    - The **TransE** embedding technique is used to map the medical concepts (head, relation, and tail entities) and align them with token embeddings generated by the LLM.
3. **Revise Action**:
    
    - If the generated triplets are incorrect, they are revised using additional knowledge from the KG. The head and tail entities are adjusted, and the triplet is re-evaluated through the Review action.
4. **Answer Action**:
    
    - Once the triplets are validated, the LLM uses them to select or generate the correct answer to the medical query.

The KGAREVION model dynamically adapts its reasoning approach based on the complexity of the question, making it suitable for both multi-choice and open-ended QA settings.

#### Training & Technology

The experiments were conducted using **NVIDIA GPUs**. The exact number and type of GPUs, as well as hyperparameters like learning rate and batch size, are not specified in detail within the paper.

KGAREVION was fine-tuned using **knowledge completion tasks**. The LLM was trained to align structured knowledge from the KG with non-structured knowledge from the LLM to improve the accuracy of generated triplets.

The **LoRA** (Low-Rank Adaptation) technique was employed to fine-tune specific parts of the LLM. The model was optimized using a next-token prediction loss during training.


------------------------------------

### Findings

- KGAREVION improved accuracy by 5.2% over 15 models in handling complex medical questions, specifically surpassing multiple baselines across different medical datasets.

- On the newly introduced MedDDx datasets (focused on differential diagnosis), KGAREVION showed up to a 10.4% increase in accuracy.

- KGAREVION’s use of generated triplets, verified against a grounded KG, proved effective in filtering out incorrect information, resulting in higher accuracy than retrieval-augmented generation (RAG)-based models.

- The dynamic adjustment of reasoning strategies according to question complexity led to significant improvements on both **multi-choice and open-ended QA tasks**.

- KGAREVION performed well across basic, intermediate, and expert difficulty levels in the MedDDx datasets. It showed a notable ability to manage more semantically complex questions, outperforming other models, especially on harder expert-level questions.

- KGAREVION showed strong performance in both multi-choice and open-ended question-answering scenarios. However, open-ended questions benefited particularly from the multi-step verification and refinement process, improving answer quality.

- KGAREVION maintained stable performance even as the number of medical concepts involved in questions increased, especially in complex cases involving multiple interacting medical concepts.

------------------------------------

### Discussion

The combination of **LLM-generated triplets and KG verification** played a crucial role in improving the system's accuracy, particularly for complex, knowledge-intensive medical queries. This approach allowed for the filtering of erroneous information, ensuring that only valid triplets contributed to the final answers.

KGAREVION’s ability to dynamically **adjust its reasoning approach based on the question’s complexity** allowed it to handle more intricate medical questions better than static methods. This adaptability is especially important in differential diagnosis scenarios, where multiple concepts must be considered simultaneously.

Unlike traditional RAG models that rely heavily on the quality of retrieved information, KGAREVION’s post-retrieval verification with knowledge graphs **reduced the reliance on potentially incomplete or inaccurate data**, contributing to its overall superior performance.

While KGAREVION showed strong improvements in accuracy, **semantically similar answer choices still posed challenges**, particularly in expert-level questions. This indicates the need for further refinement in distinguishing between closely related medical concepts.

The robustness of KGAREVION across different datasets, LLMs, and knowledge graphs suggests its **applicability to a wide range of medical question-answering scenarios**. It can be used effectively in both diagnostic settings and broader medical research contexts.

Further enhancements could focus on refining the model's ability to **handle semantically similar concepts and optimizing the verification and refinement process** for more complex datasets. Additionally, integrating real-time clinical data into the model could extend its practical utility in healthcare.

------------------------------------

### Remarks & Limitations

KGAREVION’s performance is strongly tied to the **quality and completeness of the underlying knowledge graph**. If the knowledge graph lacks key medical concepts or relationships, the system may generate incomplete or inaccurate triplets. This reliance means that in fields where the KG is not fully developed, the model’s accuracy could suffer.

While KGAREVION performs well in most scenarios, it struggles when faced with **semantically similar answer choices**. In expert-level questions, where subtle differences between medical terms or conditions are critical, the model may incorrectly filter or retain triplets, leading to wrong answers. This limitation suggests that the system requires further refinement in differentiating closely related medical concepts.

Although KGAREVION is designed for medical question answering, its **reliance on medical-specific knowledge graphs** may limit its scalability to other domains. Adapting the model to other fields would require the development of new, domain-specific knowledge graphs, which can be time-consuming and resource-intensive.

KGAREVION’s **multi-step process (triplet generation, verification, and refinement)** is computationally expensive, especially when handling complex, multi-concept questions. This can slow down performance in real-time applications, making it less suitable for scenarios requiring immediate responses, such as clinical diagnostics.

The system may struggle with medical queries involving rare diseases or novel treatments **not well-represented in the knowledge graph or training data**. Since KGAREVION relies on existing knowledge for triplet generation and verification, questions involving newly discovered concepts may not be answered accurately.

KGAREVION’s effectiveness partly depends on the **underlying large language model (LLM)**. While it significantly enhances LLM performance, the system can still inherit some of the LLM's limitations, such as biases in the pre-trained data or difficulties with highly nuanced medical language.

In open-ended question-answering tasks, KGAREVION sometimes **struggles with ambiguity**, particularly if the input query is vague or lacks sufficient context. In such cases, the generated triplets may not fully capture the question's intent, leading to incomplete or off-target answers.

Future versions of KGAREVION could **integrate real-time clinical data or electronic health records (EHRs) to enhance diagnostic capabilities**. Additionally, incorporating active learning mechanisms to continuously update the knowledge graph with new medical findings could help mitigate the issue of incomplete knowledge.

KGAREVION's implementation in healthcare settings requires careful consideration of **ethical implications**, particularly when making recommendations for treatment or diagnosis. The system should be used as a supportive tool rather than a standalone decision-maker, and continuous oversight by medical professionals is necessary to prevent the propagation of incorrect medical advice.

------------------------------------

### Citation

```
@misc{su2024knowledgegraphbasedagent,
      title={Knowledge Graph Based Agent for Complex, Knowledge-Intensive QA in Medicine}, 
      author={Xiaorui Su and Yibo Wang and Shanghua Gao and Xiaolong Liu and Valentina Giunchiglia and Djork-Arné Clevert and Marinka Zitnik},
      year={2024},
      eprint={2410.04660},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2410.04660}, 
}
```