---
title: "Eulerian at BioLaySumm: Preprocessing Over Abstract is All You Need"
authors: Satyam Modi, T Karthikeyan
year: 2024
database: ACL Anthology
citekey: modi-karthikeyan-2024-eulerian
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
  - PoA
  - flan/t5/small
  - flan/t5/base
url: https://aclanthology.org/2024.bionlp-1.77/
file: "[[Eulerian at BioLaySumm - Preprocessing Over Abstract is All You Need.pdf]]"
---

>[!title]
Eulerian at BioLaySumm: Preprocessing Over Abstract is All You Need

>[!year]
2024

>[!author]
Satyam Modi, T Karthikeyan


------------------------------------

### Summary

- This paper presents a method for **biomedical lay summarization** with a focus on improving **factuality** and readability through **preprocessing techniques** and **fine-tuning language models**.

- The authors aimed to simplify complex biomedical research articles for non-expert audiences, enhancing their accessibility and understanding.

- They employed a **Preprocessing over Abstract (PoA)** technique to improve readability by extracting and cleaning sentences from the abstract while removing supplementary details like those found in parentheses and brackets.

- For the summarization task, the team fine-tuned the **Flan-T5 model** in both **small** and **base** versions. The models were trained on two datasets: **PLOS** and **eLife**, which contain biomedical articles and corresponding lay summaries.

- The authors explored the effect of different training setups, including the use of a **cosine learning rate scheduler** and the combination of datasets to improve model generalization.

- Although the **PoA technique** showed strong performance in relevance and factuality, outperforming the Flan-T5 models in some aspects, the authors focused on refining the **Flan-T5 base model** with **PoA** for generating summaries.

- The submission achieved **1st place in factuality** and ranked **10th overall** in the **BioLaySumm 2024 shared task**, demonstrating the effectiveness of their approach in ensuring accurate and understandable summaries.

------------------------------------

### Research question

How can large language models (LLMs) be adapted and fine-tuned to generate accurate, relevant, and readable lay summaries of biomedical research articles for non-expert readers?

------------------------------------

### Methodology

#### Datasets

The study utilized two main datasets for training and evaluation: **PLOS** and **eLife**. The **PLOS dataset** is derived from the Public Library of Science and includes 24,773 instances for training and 1,376 instances for validation. The **eLife dataset** is derived from peer-reviewed articles published in the eLife journal, comprising 4,346 instances for training and 241 for validation. Both datasets focus on biomedical research articles and their lay summaries. The test set includes 142 articles from each dataset, used to evaluate the models' performance on generating summaries for lay audiences​.

#### Evaluation Metrics

The performance of the models was evaluated across three dimensions: **Relevance**, **Readability**, and **Factuality**.

- **Relevance**: Measured using **ROUGE-1**, **ROUGE-2**, **ROUGE-L**, and **BERTScore** metrics.
- **Readability**: Assessed using the **Flesch-Kincaid Grade Level (FKGL)**, **Coleman-Liau Index (CLI)**, **Dale-Chall Readability Score (DCRS)**, and **LENS** metrics. Notably, lower FKGL, CLI, and DCRS scores indicate improved readability, while higher LENS scores reflect better readability.
- **Factuality**: Evaluated using **AlignScore** and **SummaC**, two metrics designed to measure the factual consistency of the generated summaries.

#### Preprocessing

The core preprocessing step is referred to as **PoA (Preprocessing over Abstract)**. This involves extracting the initial sentences from the research papers, particularly focusing on the abstract. The abstract is regarded as a concise summary of the study, and this extraction forms the input for the summarization task. Further preprocessing involves the removal of content enclosed in parentheses, braces, and brackets, as these segments usually contain supplementary or non-essential information. The goal of this preprocessing step is to enhance readability while retaining the core message of the article​

#### Models

The summarization task was treated as a sequence-to-sequence problem, and different variants of the **Flan-T5** model were fine-tuned for this purpose.

- **Flan-T5 Small**: Initially, the Flan-T5 small model was fine-tuned using the **PLOS dataset**. The model’s input consisted of preprocessed abstracts (PoA), and the output was the corresponding lay summary. Later, the training data was expanded to include both **PLOS** and **eLife** datasets to improve the model's generalizability and robustness.
    
- **Flan-T5 Base**: The larger Flan-T5 base model was also fine-tuned using the combined **PLOS** and **eLife** datasets. It was hypothesized that the larger model would capture more complex patterns in the data, leading to better summarization performance

The fine-tuning process for the Flan-T5 base model was enhanced by introducing a **cosine learning rate scheduler**. This scheduler adjusts the learning rate dynamically, gradually reducing it during training to avoid overfitting and improve generalization.

### Training and Technology

The experiments were conducted using a **single NVIDIA A100 40GB GPU** for a total of **25 epochs**. The models were trained with a batch size of 25, a maximum input token length of 512, and a maximum output token length of 300. The learning rate was set to **1e-3** for the fine-tuning process​.

------------------------------------

### Findings

- The **Flan-T5 models** fine-tuned by the team showed notable results, particularly with the **base model** performing better on **relevance** and **readability** metrics compared to the smaller version.

- The use of **combined datasets** (PLOS and eLife) improved model performance across different metrics, although it came at the expense of **factuality**, with decreases in the **AlignScore** and **SummaC** metrics.

- Implementing a **cosine learning rate scheduler** during fine-tuning improved **readability** and **factuality** metrics, balancing the model’s performance compared to other learning rate strategies.

- The **Preprocessing over Abstract (PoA)** technique, despite its simplicity, outperformed all fine-tuned Flan-T5 models on **factuality** and **relevance**, highlighting the importance of preprocessing.

- The **Flan-T5 base model** with **PoA** demonstrated strong overall performance in factuality, which was critical for the task.

- The submission ranked **1st place in factuality** but achieved **10th place overall** in the **BioLaySumm 2024 shared task**, showing its strength in maintaining factual consistency in lay summaries.

------------------------------------

### Discussion

The results underscore the importance of **preprocessing techniques** and **fine-tuning strategies** when generating lay summaries for biomedical research. The **Preprocessing over Abstract (PoA)** method proved to be highly effective, outperforming more sophisticated models in factuality and relevance. This suggests that even simple preprocessing steps can significantly enhance summarization quality by distilling the core information and eliminating unnecessary details.

The superior performance of the **Flan-T5 base model** in relevance and readability metrics demonstrates that a larger, more complex model can better capture the nuances of biomedical texts. However, the trade-off in **factuality** metrics indicates that improving one aspect of summarization (e.g., relevance) may come at the cost of another (e.g., factual accuracy). This highlights the need for more balanced approaches, especially in fields like biomedical research where **factual consistency** is critical.

The implementation of a **cosine learning rate scheduler** was another key factor in improving the overall performance of the base model, suggesting that dynamic learning rate adjustments help in fine-tuning large models more effectively. By preventing overfitting and ensuring better generalization, the scheduler contributed to better **factual accuracy** and **readability**, making it a valuable technique for future iterations.

Interestingly, the success of the **PoA technique** without model training suggests that preprocessing alone can be an effective strategy for generating high-quality summaries, particularly when factuality is a primary concern. However, the method’s reliance on abstracts and initial sections may limit its ability to capture the full context of an article, particularly in cases where crucial information is dispersed throughout the text.

While the team’s submission performed exceptionally well in **factuality**, earning **1st place** in this metric, it ranked **10th overall**, reflecting the challenges of balancing **factuality**, **relevance**, and **readability** in summarization tasks. Future work may focus on refining models to address this balance, ensuring that summaries are not only accurate but also engaging and easy to understand for lay audiences.

In conclusion, the findings emphasize the critical role of **preprocessing** and **dynamic learning strategies** in enhancing biomedical summarization. The results suggest that even simple techniques like PoA can be highly effective in generating factual summaries, while more complex methods involving large language models need careful tuning to achieve a balance across multiple metrics. Future research could explore integrating these approaches more effectively to optimize both the factuality and overall quality of lay summaries.

------------------------------------

### Remarks & Limitations

- The **Preprocessing over Abstract (PoA)** technique demonstrated strong effectiveness in enhancing factuality and relevance, yet its simplicity may overlook critical information found beyond the abstract, potentially limiting the comprehensiveness of the summaries.

- The experiments primarily focused on fine-tuning **Flan-T5** models, which may restrict the exploration of other advanced models, such as autoregressive large language models like **LLaMA 3**. Future research could benefit from investigating different model architectures to improve summarization outcomes.

- Computational constraints impacted the experiments, potentially limiting the number of epochs and batch sizes during fine-tuning. This may affect the optimal performance of the models, suggesting that further studies with more computational resources could yield better results.

- The combination of the **PLOS** and **eLife** datasets for training might dilute the unique characteristics of each dataset. Future work should consider training separate models tailored to the specific nuances of each dataset, which could enhance the overall summarization effectiveness.

- While the models achieved high scores in factuality, the reliance on preprocessing techniques raises questions about the generalizability of these results across other biomedical fields. Summarization techniques may need to be adapted for different domains to ensure accuracy and readability.

- The paper did not explore **prompt tuning** and its potential impact on the performance of summarization tasks. Future research could investigate how prompt engineering may enhance the summarization quality, particularly for specialized biomedical contexts.

- Although the results are promising, the performance metrics and evaluation frameworks used in this study may require further development to capture user satisfaction and practical utility, which are essential for the effective dissemination of biomedical knowledge to lay audiences.

- The paper does not provide a detailed explanation of the **PoA** technique's sentence extraction process, leaving ambiguity about how the specific sentences are chosen from the abstract and other sections of the research paper.
    
- The definition of "without PoA" is unclear in the paper, as it does not explicitly describe what happens when the PoA method is not applied, making it difficult to interpret the exact nature of the comparison between "with PoA" and "without PoA" results.

- The paper does not provide the specific **prompt templates** used during the model training and evaluation phases, which limits reproducibility and the ability to fully understand the context in which the models were applied.

- It is not explicitly clear **which model** was ultimately submitted for the competition, as the paper describes various experiments with different configurations (Flan-T5 small/base, with/without cosine scheduler, and with/without PoA) without specifying the final submission model.

------------------------------------

### Citation

```
@inproceedings{modi-karthikeyan-2024-eulerian,
    title = "Eulerian at {B}io{L}ay{S}umm: Preprocessing Over Abstract is All You Need",
    author = "Modi, Satyam  and
      Karthikeyan, T",
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
    url = "https://aclanthology.org/2024.bionlp-1.77",
    pages = "826--830",
    abstract = "In this paper, we present our approach to the BioLaySumm 2024 Shared Task on Lay Sum- marization of Biomedical Research Articles at BioNLP workshop 2024. The task aims to generate lay summaries from the abstract and main texts of biomedical research articles, making them understandable to lay audiences. We used some preprocessing techniques and finetuned FLAN-T5 models for the summarization task. Our method achieved an AlignScore of 0.9914 and a SummaC metric score of 0.944.",
}
```