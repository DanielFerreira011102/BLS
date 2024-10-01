---
title: "Ctyun AI at BioLaySumm: Enhancing Lay Summaries of Biomedical Articles Through Large Language Models and Data Augmentation"
authors: Siyu Bao, Ruijing Zhao, Siqin Zhang, Jinghui Zhang, Weiyin Wang, Yunian Ru
year: 2024
database: ACL Anthology
citekey: bao-etal-2024-ctyun
tags:
  - BioLaySumm/2024
  - PLOS
  - eLife
  - ROUGE
  - BERTScore
  - FKGL
  - DCRS
  - CLI
  - LENS
  - AlignScore
  - SummaC
  - Llama/3/8B/Instruct
  - Mistral/7B/Instruct/0/2
  - Qwen/1/5/14B/Chat
  - Hard_Truncation
  - Text_Chunking
  - Fine-tuning
url: https://aclanthology.org/2024.bionlp-1.79/
file: "[[Ctyun AI at BioLaySumm - Enhancing Lay Summaries of Biomedical Articles Through Large Language Models and Data Augmentation.pdf]]"
---

>[!title]
Ctyun AI at BioLaySumm: Enhancing Lay Summaries of Biomedical Articles Through Large Language Models and Data Augmentation

>[!year]
2024

>[!author]
Siyu Bao, Ruijing Zhao, Siqin Zhang, Jinghui Zhang, Weiyin Wang, Yunian Ru


------------------------------------

### Summary
1. 

------------------------------------

### Research question

How can large language models (LLMs) be adapted and fine-tuned to generate accurate, relevant, and readable lay summaries of biomedical research articles for non-expert readers?

------------------------------------

### Methodology

#### Datasets

The models are trained and evaluated on two biomedical datasets, **eLife** and **PLOS**, which include full-text articles along with their corresponding lay summaries. These datasets are designed to create accessible summaries for non-expert readers. The **eLife** dataset provides lay summaries written by expert editors, whereas the **PLOS** dataset contains lay summaries written by the article authors. The size of the datasets varies, with **eLife** containing 4,346 training instances and 241 validation instances, and **PLOS** including 24,773 training instances and 1,376 validation instances. The average token length is significantly larger for the eLife articles (16,555 tokens) compared to PLOS articles (10,289 tokens), making preprocessing necessary to handle lengthy inputs.

#### Evaluation Metrics

To assess the quality of the generated lay summaries, several automatic metrics were used to measure relevance, readability, and factual accuracy:

- **Relevance** was measured using **ROUGE (1, 2, and L)** and **BERTScore**.
- **Readability** was evaluated through the **Flesch-Kincaid Grade Level (FKGL)**, **Dale-Chall Readability Score (DCRS)**, **Coleman-Liau Index (CLI)**, and **Learnable Evaluation Metric for Simplification (LENS)**. Lower FKGL, DCRS, and CLI scores indicate better readability.
- **Factuality** was measured using **AlignScore** and **SummaC**, ensuring consistency between the generated summaries and the source content​.

#### Preprocessing

Given the constraint on input length for large language models, two approaches were adopted to handle long articles:

1. **Hard Truncation**: This method retained only the first 15,000 tokens of each article, leveraging the fact that important information is usually presented early in the articles. This approach ensured that critical content was preserved while fitting within the token limit.
    
2. **Text Chunking**: For articles longer than 15,000 tokens, the text was split into chunks of up to 15,000 tokens. Summaries were generated for each chunk individually, and these chunk-level summaries were later combined to create a final summary.

#### Data Augmentation

To address the discrepancies between the chunked articles and the corresponding full-text lay summaries, **data augmentation** was employed using the **Mixtral 8x7B model**. The Mixtral model generated summaries for the article fragments, which were used as inputs during training, with the original lay summary serving as the target output. This ensured that all parts of the article were represented in the model training process, minimizing information loss from chunking​.

#### Prompt Engineering

Different prompts were designed for the Hard Truncation and Text Chunking methods to guide the summarization process:

- For **unmodified articles** (those with fewer than 15,000 tokens), a straightforward prompt was used to generate summaries.
- For **chunked articles**, the prompts provided chunk-specific instructions, directing the model to generate concise summaries for each fragment.
- An **aggregated summary prompt** was applied to merge chunk-level summaries into a coherent whole, instructing the model to condense the text into a final lay summary​

------------------------------------

### Findings
1. 

------------------------------------

### Discussion
1. 

------------------------------------

### Remarks & Limitations
1. 

------------------------------------

### Citation

```
@inproceedings{bao-etal-2024-ctyun,
    title = "Ctyun {AI} at {B}io{L}ay{S}umm: Enhancing Lay Summaries of Biomedical Articles Through Large Language Models and Data Augmentation",
    author = "Bao, Siyu  and
      Zhao, Ruijing  and
      Zhang, Siqin  and
      Zhang, Jinghui  and
      Wang, Weiyin  and
      Ru, Yunian",
    editor = "Demner-Fushman, Dina  and
      Ananiadou, Sophia  and
      Miwa, Makoto  and
      Roberts, Kirk  and
      Tsujii, Junichi",
    booktitle = "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.bionlp-1.79",
    pages = "837--844",
    abstract = "Lay summaries play a crucial role in making scientific research accessible to a wider audience. However, generating lay summaries from lengthy articles poses significant challenges. We consider two approaches to address this issue: Hard Truncation, which preserves the most informative initial portion of the article, and Text Chunking, which segments articles into smaller, manageable chunks. Our workflow encompasses data preprocessing, augmentation, prompt engineering, and fine-tuning large language models. We explore the influence of pretrained model selection, inference prompt design, and hyperparameter tuning on summarization performance. Our methods demonstrate effectiveness in generating high-quality, informative lay summaries, achieving the second-best performance in the BioLaySumm shared task at BioNLP 2024.",
}
```