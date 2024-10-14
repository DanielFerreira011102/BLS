---
title: "UIUC_BioNLP at BioLaySumm: An Extract-then-Summarize Approach Augmented with Wikipedia Knowledge for Biomedical Lay Summarization"
authors: Zhiwen You, Shruthan Radhakrishna, Shufan Ming, Halil Kilicoglu
year: 2024
database: ACL Anthology
citekey: you-etal-2024-uiuc
tags:
  - BioLaySumm/2024
  - GPT/3/5
  - RAG
  - Wikipedia
  - Fine-tuning
  - LED
  - TextRank
  - BERT_Clustering
  - GPT/4
  - eLife
  - PLOS
  - Zero-shot
  - BART
  - ROUGE
  - BERTScore
  - FKGL
  - DCRS
  - CLI
  - LENS
  - AlignScore
  - SummaC
url: https://aclanthology.org/2024.bionlp-1.11/
file: "[[UIUC_BioNLP at BioLaySumm - An Extract-then-Summarize Approach.pdf]]"
---

>[!title]
UIUC_BioNLP at BioLaySumm: An Extract-then-Summarize Approach Augmented with Wikipedia Knowledge for Biomedical Lay Summarization

>[!year]
2024

>[!author]
Zhiwen You, Shruthan Radhakrishna, Shufan Ming, Halil Kilicoglu


------------------------------------

### Summary

- This paper describes the development of two distinct approaches to **biomedical lay summarization** through an **extract-then-summarize** framework to reduce input length while maintaining summarization quality.
- The goal is to simplify biomedical research articles for laypeople, making scientific content accessible. 
- The two models utilized in the task were **fine-tuned GPT-3.5** and **LED (Longformer Encoder-Decoder)**, each processed and optimized differently for summarization tasks. 
- The paper explores the effectiveness of using **unsupervised extractive summarization techniques**, including **TextRank** and **BERT-based clustering**, to condense the content, which is further refined with **retrieval-augmented generation (RAG)**, integrating external information from **Wikipedia**. 
- The models were tested on two datasets, **eLife** and **PLOS**, which contain biomedical articles and corresponding lay summaries.
- The paper likely explores **post-processing with GPT-4** to strike a better balance between **factuality and readability**.
- The fine-tuned GPT-3.5 model achieved the highest overall ranking and demonstrated the best relevance performance in the **BioLaySumm 2024 shared task**.

------------------------------------

### Research question

How can large language models (LLMs) be adapted and fine-tuned to generate accurate, relevant, and readable lay summaries of biomedical research articles for non-expert readers?

------------------------------------

### Context

The paper was created in the context of the 23rd Workshop on Biomedical Language Processing (BioNLP 2024), which included a shared task called BioLaySumm. The task addresses the growing challenge of making biomedical research accessible to laypeople. With the rapid increase in scientific publications, particularly in the biomedical field, lay audiences struggle to keep up due to the technical complexity of these studies. Therefore, lay summarization—a form of text summarization that focuses on making content more understandable for non-experts—is needed.

Although summarization of scientific papers has been widely studied, creating summaries specifically tailored to non-expert audiences is a relatively new and critical area of focus, especially given the importance of biomedical research in improving public health.

------------------------------------

### Methodology

![[be5076402f95ccee97f4f3680ab2e6.png|center]]

#### Datasets

The models are trained and evaluated on two key biomedical datasets, **eLife** and **Public Library of Science (PLOS)**, which provide full-text biomedical articles along with their corresponding lay summaries. These datasets focus on creating summaries for laypeople. The **eLife** dataset features lay summaries crafted by expert editors, while **PLOS** includes lay summaries written by the article authors. Both datasets cover a broad range of topics within life sciences and medicine. The average token lengths are 13,000 for eLife articles and 9,000 for PLOS articles, respectively​.

#### Evaluation Metrics

The models performance was evaluated using various automatic metrics related to relevance, readability, and factuality. Relevance is measured through **Recall-Oriented Understudy for Gisting Evaluation or ROUGE (1, 2, and L)** and **BERTScore**. Readability metrics include the **FleschKincaid Grade Level (FKGL)**, **Dale-Chall Readability Score (DCRS)**, **Coleman-Liau Index (CLI)**, and **Learnable Evaluation Metric for Simplification (LENS)**. Notably, lower FKGL, DCRS, and CLI scores signify improved readability. Factuality evaluation incorporates **AlignScore** and **SummaC**.
#### Preprocessing

Before training the models, preprocessing steps were applied to reduce the length of input articles. This involved two key methods:

1. **Section Reordering**: Articles were restructured based on relevance to summary content, with sections ordered as abstract, background, conclusions, results, and methods. This was based on cosine similarity between sections and lay summaries​.
2. **Unsupervised Extractive Summarization**: Two extractive summarization approaches were used to condense articles:
    - **TextRank**: A graph-based ranking algorithm was used to identify key sentences within the text.
    - **BERT-based Clustering**: PubMedBERT embeddings were applied to cluster sentences by similarity, selecting those closest to the cluster centroids as the most relevant​.

#### Model 1: Longformer Encoder-Decoder (LED)

The **LED model** is a transformer-based architecture designed for long document summarization. Given the article length, the authors used the **LED-base model** with a maximum input length of 8,192 tokens, fine-tuning it on reordered article sections and the extracted summaries. For knowledge augmentation, **dense retrieval (DPR)** was employed to pull relevant passages from a Wikipedia corpus, which were then combined with the extracted summaries to enrich the input for summarization​. The model was optimized using **cross-entropy loss** to enhance its performance.
#### Model 2: GPT-3.5

The **GPT-3.5 model** was fine-tuned using an extract-then-summarize approach. The fine-tuning process involved feeding **TextRank-extracted sentences** into GPT-3.5 to generate summaries. Due to memory constraints, only 40 sentences from each article were included in the input. The model was fine-tuned separately for each dataset, with the number of training examples ranging from 100 to 400 articles​. The fine-tuning utilized a **contrastive loss** function, which helped improve the model’s ability to align extracted sentences with lay summaries.

#### Postprocessing

To improve factual accuracy, **GPT-4** was used to refine the generated lay summaries. This involved prompting GPT-4 with both the article’s abstract and the generated summary, instructing it to enhance factual consistency by aligning the summary more closely with the abstract.

Despite this attempt, the results showed minimal improvement in factuality (measured by **AlignScore**) and no significant gains in readability or relevance. In fact, readability scores slightly declined after using GPT-4 for postprocessing.

#### Training & Technology

Both models were trained using a **single NVIDIA Tesla V100 GPU** with 32 GB memory. The LED model was trained for **one epoch** with a batch size of 4, using **Adam optimization**. The **GPT-3.5** fine-tuning process leveraged the **OpenAI API** for model adjustments​.

------------------------------------

### Findings

- The **GPT-3.5 model** outperformed others in the **BioLaySumm 2024 shared task**, particularly excelling in **relevance**, with summaries closely matching the original biomedical content.
- The **Longformer Encoder-Decoder (LED)** model also showed strong performance but faced issues with **factual consistency** due to the use of external data from **Wikipedia**, which sometimes introduced irrelevant or incorrect information.
- **Section reordering** (prioritizing abstract, background, conclusions, results, and methods) helped improve summary relevance by aligning the structure with lay audiences' needs.
- **Extractive summarization techniques** (TextRank and BERT-based clustering) effectively reduced article length while maintaining key information, allowing for more efficient and relevant summaries.
- Larger models, such as **PubMed LED**, improved relevance but reduced **readability**, making them less suitable for lay audiences. In contrast, smaller models like **GPT-3.5** balanced **relevance** and **readability** more effectively.
- Both models occasionally included **irrelevant information**, with **GPT-3.5** sometimes generating overly long summaries that introduced extraneous content not present in the original article, and **LED** suffering from factual inconsistencies due to noise in the external data.
- The abstract of a biomedical article often achieves higher factuality scores than the gold lay summary. Lay summaries are often more interpretative and simplified, which can lead to lower factuality scores even though they are still accurate.
- Despite its advanced language capabilities, GPT-4 post-processing did not lead to a noticeable improvement in aligning lay summaries with abstracts.

------------------------------------

### Discussion

The results of the BioLaySumm 2024 shared task highlight several important implications for the development of automated summarization models tailored to biomedical content for lay audiences. The **GPT-3.5** model, through its fine-tuning process, showed remarkable improvement in relevance metrics, suggesting that LLMs, when optimized with the right extractive techniques, can generate summaries that are not only relevant but also more accessible to non-expert audiences.

While the **LED model** benefited from the integration of external knowledge via Wikipedia, the added information occasionally affected the factuality of the summaries. This raises a broader concern about the trade-offs between including external knowledge to enhance summaries and ensuring that such information remains accurate and relevant.

The study reflects on the trade-off between model size and readability, noting that while larger models (like PubMed LED) tend to improve relevance, they often reduce readability, a key factor when targeting non-expert audiences. This underscores the need for fine-tuning model size and complexity based on the desired output, particularly in specialized tasks like biomedical summarization.

The positive impact of section reordering suggests that structural considerations can play a significant role in improving the clarity and relevance of summaries for lay readers. By prioritizing sections like conclusions and results, the summarization process aligns more closely with how non-experts process information, emphasizing outcomes and implications over technical details. This method could serve as a guideline for future summarization tasks, particularly in specialized fields like biomedicine. 

Traditional extractive methods, such as TextRank and BERT-based clustering, have proven useful in summarizing content while retaining key information. However, these methods can be rigid and may struggle to provide clear, easy-to-understand explanations like abstractive models do. Extractive approaches also run the risk of oversimplifying or missing important details, which are crucial for understanding complex topics like biomedical research. On the other hand, generative models are more flexible and can create new, more accessible explanations, but they often face challenges with accuracy and can sometimes introduce unnecessary information.

Overall, the task results suggest that achieving an optimal balance between relevance, readability, and factual consistency remains a core challenge in biomedical text summarization. While models like GPT-3.5 show promise, further exploration of hybrid approaches, leveraging both extractive and abstractive techniques, may offer a pathway to refining future models. The integration of adaptive strategies, including dynamic reordering and context-sensitive filtering of external data, is also crucial for ensuring that summaries are both informative and accessible to lay audiences.

------------------------------------

### Remarks & Limitations

- The **extract-then-summarize framework** proved efficient for summarizing long biomedical documents, enabling the models to handle input length constraints while maintaining performance.
- GPT-3.5 relied solely on **extractive summaries** for fine-tuning, whereas the LED model utilized a combination of article sections ranked for relevance, RAG for external knowledge from Wikipedia, and extractive summaries.
- Fine-tuning **GPT-3.5** with contrastive learning allowed it to produce more accurate and informative lay summaries than LED.
- The fine-tuning experiments were constrained by computational resources, limiting the number of epochs and batch sizes for some models (e.g., LED was fine-tuned for only **one epoch** due to resource limits).
- Despite the improvements in relevance and readability, the use of RAG occasionally introduced **inconsistent or noisy data**, which negatively impacted the factuality of the generated summaries. Future work could explore improving the factuality of model-generated summaries, particularly by refining the retrieval-augmented generation approach and developing **domain-specific extractors** to better tailor summaries for lay audiences.
- The eLife and PLOS datasets focus primarily on life sciences, which may limit the generalizability of the results across other biomedical fields.
- While GPT-3.5 performed well in this task, the closed-source nature of the model may limit reproducibility and future research, particularly as it undergoes updates without versioning.

------------------------------------

### Citation

```
@inproceedings{you-etal-2024-uiuc,
    title = "{UIUC}{\_}{B}io{NLP} at {B}io{L}ay{S}umm: An Extract-then-Summarize Approach Augmented with {W}ikipedia Knowledge for Biomedical Lay Summarization",
    author = "You, Zhiwen  and
      Radhakrishna, Shruthan  and
      Ming, Shufan  and
      Kilicoglu, Halil",
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
    url = "https://aclanthology.org/2024.bionlp-1.11",
    pages = "132--143",
    abstract = "As the number of scientific publications is growing at a rapid pace, it is difficult for laypeople to keep track of and understand the latest scientific advances, especially in the biomedical domain. While the summarization of scientific publications has been widely studied, research on summarization targeting laypeople has remained scarce. In this study, considering the lengthy input of biomedical articles, we have developed a lay summarization system through an extract-then-summarize framework with large language models (LLMs) to summarize biomedical articles for laypeople. Using a fine-tuned GPT-3.5 model, our approach achieves the highest overall ranking and demonstrates the best relevance performance in the BioLaySumm 2024 shared task.",
}
```