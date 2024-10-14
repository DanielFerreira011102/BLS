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

The complexity of biomedical knowledge presents unique challenges for artificial intelligence (AI) systems, especially in the field of medicine. Unlike other scientific disciplines, such as physics or chemistry, **medical reasoning often involves analogy-based thinking and relies heavily on nuanced, context-specific information**. Medical professionals must draw on various reasoning strategies to diagnose diseases, prescribe treatments, and understand the mechanisms behind illnesses. For instance, models in medical research frequently involve using analogies between organisms to understand disease mechanisms that apply to humans.

This creates a need for AI systems that can handle **complex, knowledge-intensive queries** in medicine, where simple rule-based or fact-retrieval approaches are insufficient. Large language models (LLMs) like GPT-4 have demonstrated remarkable success in general-purpose tasks but struggle with the **highly specialized knowledge** required in the medical domain. These models may retrieve incorrect or incomplete information and fail to account for the specific medical context, such as patient demographics, disease-specific characteristics, or localized medical conditions.

Furthermore, existing models face challenges in integrating multiple types of evidence, including structured scientific knowledge from research studies and **tacit expert knowledge** based on real-world medical practice. This gap highlights the need for a more robust approach, combining the **structured knowledge** of medical concepts with the **flexibility** and **reasoning capabilities** of LLMs.

**Knowledge graphs (KGs)**, which represent complex relationships between medical concepts, have emerged as a valuable resource in AI for healthcare. However, existing systems that rely solely on KGs often lack the depth and adaptability to tackle complex medical reasoning, particularly when the knowledge in these graphs is incomplete or the reasoning requires integrating multiple medical concepts.

To address these challenges, the paper introduces **KGAREVION**, a new AI agent that uses both LLMs and KGs to answer complex medical questions accurately and flexibly. This agent was developed to improve upon previous models by grounding LLM-generated information in structured medical knowledge, ensuring accuracy and relevance in contexts where medical reasoning is especially difficult. This work came at a time when AI in healthcare was advancing rapidly, but still grappling with integrating complex medical knowledge into AI-driven solutions for real-world applications.

------------------------------------

### Methodology

The paper outlines the design and functioning of **KGAREVION**, an AI model designed to answer complex, knowledge-intensive medical questions. The methodology involves several key steps, from gathering datasets to training and evaluating the model.

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

------------------------------------

### Findings


------------------------------------

### Discussion


------------------------------------

### Remarks & Limitations


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