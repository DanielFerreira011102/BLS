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
  - SFT
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

- This paper presents **Ctyun AI's** approach to generating **lay summaries** for biomedical articles by employing **large language models (LLMs)** and **data augmentation** techniques. 

- The primary aim is to make complex biomedical research more accessible to non-experts.
    
- Two preprocessing methods are introduced to manage lengthy articles: **Hard Truncation**, which retains the first 15,000 tokens of an article, and **Text Chunking**, which divides the article into chunks of up to 15,000 tokens. These methods address the input length limitations of large models while preserving critical information.
    
- To further enhance the summarization process, **data augmentation** was used. This involved generating summaries for chunked text using the **Mixtral 8x7B** model, ensuring that the training dataset covered all parts of the article and improved the quality of the generated lay summaries.
    
- The experiments focused on **fine-tuning** three pretrained LLMs: **Qwen1.5-14B-Chat**, **Mistral-7B-Instruct-v0.2**, and **Meta-Llama-3-8B-Instruct**. Among them, **Mistral-7B-Instruct-v0.2** was selected for further experimentation due to its balance of performance across relevance, readability, and factuality metrics.
    
- The approach yielded strong performance in the **BioLaySumm 2024 shared task**, achieving **second place overall**. The paper highlights the importance of combining models fine-tuned on different datasets (eLife and PLOS) and using data augmentation to handle long and complex articles more effectively.

------------------------------------

### Research question

How can large language models (LLMs) be adapted and fine-tuned to generate accurate, relevant, and readable lay summaries of biomedical research articles for non-expert readers?

------------------------------------

### Methodology


![[1b0828750de2a5edb51f9dc99c6342.png|center]]

#### #### Datasets

The models in this study were trained and evaluated on two biomedical datasets, **eLife** and **PLOS**, both containing full-text biomedical articles along with their corresponding lay summaries. These datasets are designed for generating lay summaries aimed at non-expert readers:

- **eLife**: 4,346 training instances and 241 validation instances, with an average token length of 16,555 tokens per article.
- **PLOS**: 24,773 training instances and 1,376 validation instances, with an average token length of 10,289 tokens per article.

Due to the length of these articles, preprocessing techniques were essential to reduce the input size to fit within the token limits of large language models

#### Evaluation Metrics

To assess the quality of the generated lay summaries, several automatic evaluation metrics were employed:

- **Relevance**: The overlap between the generated summary and the reference summary was measured using **ROUGE (1, 2, and L)** and **BERTScore**. Higher scores indicated better relevance.
- **Readability**: The ease of understanding the generated summaries was evaluated using the **Flesch-Kincaid Grade Level (FKGL)**, **Dale-Chall Readability Score (DCRS)**, **Coleman-Liau Index (CLI)**, and **Learnable Evaluation Metric for Simplification (LENS)**. Lower scores for FKGL, DCRS, and CLI indicate better readability, while higher LENS scores suggest better simplification.
- **Factuality**: To ensure that the generated summaries were factually consistent with the source text, **AlignScore** and **SummaC** were used​.

#### Preprocessing

Given the constraint on input length for large language models, two preprocessing approaches were applied to manage lengthy articles:

1. **Hard Truncation**: This method retained the first 15,000 tokens from each article. This was based on the assumption that critical information is presented early in scientific articles. It allowed the model to process articles within the token limit while minimizing information loss from truncation.
    
2. **Text Chunking**: For articles exceeding 15,000 tokens, the text was split into chunks of 15,000 tokens or less. A summary was generated for each chunk, and these chunk-level summaries were later combined into a final lay summary​.
    

#### Data Augmentation

To bridge the gap between the chunked articles and the full-text lay summaries, **Mixtral 8x7B** was used for data augmentation. The model generated summaries for each article fragment. These fragment summaries were used during training, while the original lay summary served as the target output. This technique ensured that all parts of the article were utilized during training without losing critical information due to chunking​.

#### Prompt Engineering

Different prompts were employed to guide the model's summarization process for the two preprocessing strategies:

- **Unmodified Data**: Articles under 15,000 tokens were directly summarized using a straightforward prompt.
- **Augmented Data from Chunking**: For chunked articles, a prompt directed the model to summarize each fragment.
- **Aggregated Summary Data**: After chunk-level summaries were generated, another prompt instructed the model to merge them into a concise and coherent final summary​.

#### Models

The experiments involved three pretrained large language models:

1. **Qwen1.5-14B-Chat**
2. **Mistral-7B-Instruct-v0.2**
3. **Meta-Llama-3-8B-Instruct**

All models were fine-tuned on the **Hard Truncation** dataset, which retained the first 15,000 tokens of each article. After the initial fine-tuning, **Mistral-7B-Instruct-v0.2** was selected for subsequent experiments due to its balanced performance across relevance, readability, and factual accuracy metrics. The fine-tuning process allowed each model to adapt to the task of summarizing biomedical articles for lay audiences

#### Training & Technology

The models were fine-tuned using **Supervised Fine-Tuning (SFT)** with a **learning rate of 1e-5** and a **global batch size of 64**. The training was conducted for **one epoch** across all models during the initial fine-tuning phase.

For the **Mistral-7B-Instruct-v0.2** model, additional experiments were conducted to optimize hyperparameters. These experiments included testing two learning rates, 1e-5 and 2e-5, as well as comparing the performance of single-epoch versus dual-epoch training, both using a learning rate of 1e-5.

These experiments aimed to balance model performance and computational efficiency. During inference, different **prompts** were designed to evaluate their influence on the model's ability to generate coherent and relevant lay summaries

------------------------------------

### Findings

- **Ctyun AI** achieved strong results, **ranking second overall in BioLaySumm 2024 shared task**, largely due to its use of fine-tuned large language models (LLMs) and robust data preprocessing techniques.

- Both approaches showed strengths and weaknesses depending on the dataset. **Hard Truncation** maintained summary completeness in shorter articles but risked omitting crucial content in longer articles, while **Text Chunking** better preserved content from longer articles but introduced artificial boundaries that sometimes disrupted coherence.

- Among the models tested, **Mistral-7B-Instruct-v0.2** struck the best balance between relevance and readability, outperforming alternatives like **Qwen1.5-14B-Chat** and **Meta-Llama-3-8B-Instruct**, especially in **ROUGE and factuality metrics**.

- **Data augmentation** using **Mixtral 8x7B** helped improve summaries for fragmented articles, but this also introduced stylistic biases. These Mixtral-generated fragments may not always align with the overall summary, reducing factual consistency in some instances.

- The use of a **complex prompt** during inference significantly improved the **relevance** and **factuality** of the generated lay summaries compared to simpler prompts, although it slightly reduced **readability** scores.

- Single-epoch fine-tuning with a learning rate of 1e-5 provided optimal results. Multi-epoch fine-tuning marginally decreased overall performance by introducing slight redundancies.

- The **Hard Truncation** method performed better on the **eLife** dataset, while **Text Chunking** excelled with the **PLOS** dataset, leading the team to use an ensemble approach that combined both methods for their final submission.

- Longer biomedical articles in the dataset led to varying results. **Hard Truncation** sometimes omitted important details, while **Text Chunking** introduced coherence issues due to artificial divisions.

------------------------------------

### Discussion

The results of the experiments highlight the importance of model selection, data preprocessing strategies, and prompt engineering in generating high-quality lay summaries for biomedical articles. The combination of these elements enabled Ctyun AI to achieve competitive performance in the BioLaySumm 2024 shared task.

One of the most notable findings is the effectiveness of **Text Chunking** in handling lengthy biomedical articles. This method ensured that no crucial content was lost, especially in articles that exceeded token limits. However, the introduction of artificial boundaries between text chunks occasionally disrupted the flow and coherence of the generated summaries. This suggests that while chunking is useful for preserving content, future models might need more sophisticated techniques to maintain contextual continuity across chunks.

The **Hard Truncation** approach, while effective for shorter articles, risked omitting important details from longer ones, particularly those where significant information was presented later in the text. This limitation indicates that summarization techniques must be flexible enough to adapt to article structures where critical findings may not always be front-loaded.

In terms of **pretrained model performance**, Mistral-7B-Instruct-v0.2 demonstrated the best balance between relevance, readability, and factuality. This suggests that models with fewer parameters, if fine-tuned appropriately, can be just as effective as larger, more complex models in certain tasks. The performance of Mistral further underscores the importance of choosing a model that aligns with both the task requirements and computational constraints.

The role of **data augmentation** through Mixtral 8x7B also proved valuable, especially for ensuring that summaries of chunked fragments aligned with the full text. However, this augmentation introduced inconsistencies between fragment-based and full-text summaries, potentially reducing overall factual alignment. This points to the need for more integrated approaches where data augmentation can harmonize both the chunked and full-text contexts.

The influence of **prompt engineering** was another critical factor, with more complex prompts delivering better performance in relevance and factuality. This finding highlights the role of well-constructed prompts in steering large language models toward generating more accurate and contextually rich outputs. Nonetheless, the slight reduction in readability observed with complex prompts suggests that there may be a trade-off between achieving high relevance and maintaining user-friendly summaries.

**Hyperparameter tuning** further refined the summarization process, with single-epoch fine-tuning emerging as the optimal configuration. This finding implies that over-tuning a model may introduce unnecessary redundancy, whereas a single fine-tuning pass strikes the right balance between performance and efficiency.

Finally, the dataset-specific performance of both **Hard Truncation** and **Text Chunking** suggests that the characteristics of different biomedical datasets, such as PLOS and eLife, may require tailored summarization approaches. An ensemble approach, combining the strengths of both methods, allowed for better results overall, but it also highlighted the complexity of addressing varying dataset needs.

In summary, the findings emphasize the need for adaptable and flexible methods in generating biomedical lay summaries. Future work should focus on addressing the coherence issues posed by Text Chunking, refining data augmentation techniques to ensure consistency, and continuing to optimize the balance between relevance, factuality, and readability in generated summaries.

------------------------------------

### Remarks & Limitations

- **Text Chunking** and **Hard Truncation** allowed the models to handle lengthy biomedical articles, with Text Chunking ensuring comprehensive content coverage and Hard Truncation focusing on the most informative initial sections. However, both methods showed trade-offs, such as disrupting summary coherence with chunking or omitting important content with truncation.

- **Pretrained model performance** varied, with Mistral-7B-Instruct-v0.2 performing well in relevance and readability, but the use of larger models was limited by computational resources. Future work could explore fine-tuning larger models like GPT-4 for improved results.

- The study did not investigate the **differential impact of distinct article sections** (e.g., introductions vs. conclusions) on summary quality, despite these sections often containing crucial information. Future research should explore section-specific approaches to improve summarization performance.

- **Data augmentation** using Mixtral 8x7B improved the handling of chunked articles but introduced stylistic biases, which sometimes led to inconsistencies between chunk-based summaries and overall summaries, reducing factuality in some cases.

- The experiments were limited by **computational constraints**, restricting some models to minimal fine-tuning. More epochs and larger batch sizes could enhance both factuality and readability.

- The study focused on **eLife** and **PLOS datasets**, which are heavily centered on life sciences. The generalizability of the methods to other biomedical or interdisciplinary datasets remains uncertain.

- **Incremental pretraining** in specialized domains was not explored. Such an approach could help the model better understand complex biomedical terminology and generate more accurate, accessible summaries for lay audiences.

- **Technical terminology comprehension** remains a challenge. The model sometimes struggled to simplify complex scientific language into lay terms. Incremental pretraining could further enhance the model’s ability to convey technical terms in an accessible way.

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