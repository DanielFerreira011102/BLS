---
title: "Ctyun AI at BioLaySumm: Enhancing Lay Summaries of Biomedical Articles Through Large Language Models and Data Augmentation"
authors: Siyu Bao, Ruijing Zhao, Siqin Zhang, Jinghui Zhang, Weiyin Wang, Yunian Ru
year: 2024
database: ACL Anthology
citekey: bao-etal-2024-ctyun
tags:
  - BioLaySumm/2024
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
1. 

------------------------------------

### Methodology

#### Datasets

The model was trained and evaluated using two biomedical datasets: **PLOS** and **eLife**. These datasets consist of biomedical research articles along with their corresponding lay summaries. In the **PLOS** dataset, lay summaries are authored by the article's original authors, while **eLife** contains summaries crafted by expert editors. **PLOS** has 24,773 articles for training and 1,376 for validation, whereas **eLife** has 4,346 articles for training and 241 for validation. Articles in these datasets vary in length, with some exceeding 15,000 tokens, which posed challenges for processing​.

#### Evaluation Metrics

The performance of the models was evaluated using multiple metrics:

- **Relevance**: ROUGE (1, 2, and L) and **BERTScore** were used to assess the relevance of the generated summaries compared to the source articles.
- **Readability**: Readability was measured using **Flesch-Kincaid Grade Level (FKGL)**, **Dale-Chall Readability Score (DCRS)**, **Coleman-Liau Index (CLI)**, and **LENS**. Lower scores for FKGL, DCRS, and CLI indicated better readability, while higher LENS scores suggested better simplification.
- **Factuality**: The factual consistency between the summaries and source texts was measured using **AlignScore** and **SummaC**, where higher scores indicated better factual alignment​.

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