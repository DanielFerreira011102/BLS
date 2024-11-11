---
title: "Team YXZ at BioLaySumm: Adapting Large Language Models for Biomedical Lay Summarization"
authors: Jieli Zhou, Cheng Ye, Pengcheng Xu, Hongyi Xin
year: 2024
database: ACL Anthology
citekey: zhou-etal-2024-team
tags:
  - biolaysumm/2024
  - elife
  - plos
  - claude/3/opus
  - gemini/1/5/pro
  - gpt/4
  - llama/3/8b/instruct
  - llama/3/openbiollm/70b
  - bart
  - lora
  - beautifulsoup
  - title-infusion
  - k-shot-prompting
  - llm-rewriting
  - instruction-tuning
  - fine-tuning
  - bge/m3
  - rouge
  - bert-score
  - fkgl
  - dcrs
  - cli
  - lens
  - align-score
  - summa-c
url: https://aclanthology.org/2024.bionlp-1.76/
file: "[[Team YXZ at BioLaySumm - Adapting Large Language Models for Biomedical Lay Summarization.pdf]]"
---

>[!title]
Team YXZ at BioLaySumm: Adapting Large Language Models for Biomedical Lay Summarization

>[!year]
2024

>[!author]
Jieli Zhou, Cheng Ye, Pengcheng Xu, Hongyi Xin


------------------------------------

### Summary

- The paper describes the development of techniques for **adapting large language models (LLMs)** to generate **biomedical lay summaries** for non-expert audiences.
- The researchers aimed to optimize LLMs for summarization, specifically focusing on enhancing **readability** while balancing **factuality** and **relevance**.
- The authors tested five advanced LLMs (**Claude-3-Opus**, **Gemini-1.5-Pro**, **GPT-4**, **Llama3-8B-Instruction** and **OpenBioLLM-Llama3-70B**) and compared their results against the official baseline method based on **BART**.
- The team implemented multiple techniques, including **title infusion**, **K-shot prompting**, **LLM rewriting**, and **instruction fine-tuning**, to adapt the LLMs.
- The models were tested on two datasets, **eLife** and **PLOS**, which contain biomedical articles and corresponding lay summaries.
- The study found that, while LLMs performed well in **readability**, they often struggled with maintaining **factuality** and **relevance** compared to smaller models like **BART**.
- The team’s submission achieved **first place** in **readability** in the **BioLaySumm 2024 shared task** and was among the top teams in overall performance.

------------------------------------

### Research question

How can **large language models (LLMs)** be adapted to generate **readable**, **relevant**, and **factually consistent** lay summaries of biomedical articles for non-expert audiences?

------------------------------------

### Context

The paper was created in the context of the 23rd Workshop on Biomedical Language Processing (BioNLP 2024), which included a shared task called BioLaySumm. The task addresses the growing challenge of making biomedical research accessible to laypeople. With the rapid increase in scientific publications, particularly in the biomedical field, lay audiences struggle to keep up due to the technical complexity of these studies. Therefore, lay summarization—a form of text summarization that focuses on making content more understandable for non-experts—is needed.

Although summarization of scientific papers has been widely studied, creating summaries specifically tailored to non-expert audiences is a relatively new and critical area of focus, especially given the importance of biomedical research in improving public health.

------------------------------------

### Methodology

#### Datasets

The dataset used for this study comprises 31,020 article-summary pairs from two major biomedical journals:

- **PLOS**: The lay summaries are written by the article authors and are generally shorter, with an average length of 175.6 words.
- **eLife**: The summaries are written by journal editors and tend to be longer, averaging 347.6 words.

The dataset is split into train, validation, and test sets. The test set lay summaries are hidden for competition purposes, and models are evaluated based on their ability to generate readable, relevant, and factually accurate lay summaries from the full article text

#### Evaluation Metrics

The performance of the models is evaluated across three main categories: **relevance**, **readability**, and **factuality**.

1. **Relevance**:
    
    - **ROUGE (R1, R2, RL)**: Measures the overlap of generated summaries with reference summaries.
    - **BERTScore**: Uses embeddings to assess the semantic similarity between the generated and reference summaries.

1. **Readability**:
    - **Flesch-Kincaid Grade Level (FKGL)**, **Dale-Chall Readability Score (DCRS)**, and **Coleman-Liau Index (CLI)**: These metrics estimate the education level required to understand the summary (lower scores indicate better readability).
    - **LENS**: A learnable readability metric that aligns closely with human judgment on readability preferences.
    
1. **Factuality**:
    - **AlignScore**: Measures the consistency of the summary by checking if all information is contained in the reference.
    - **SummaC**: Uses natural language inference (NLI) to evaluate the factual consistency of the generated summary with the original text​.

#### Preprocessing

To optimize the inputs for the models, the preprocessing steps included:

1. **Title Infusion**: Since article titles were not provided in the test data, **BeautifulSoup** was used to scrape article titles from journal websites (eLife and PLOS). The titles were then integrated into the model prompts to provide a high-level context for summarization​. The authors thought that titles were essential for summarization since they encapsulate the high-level description of the articles.
2. **Abstract-based Summarization**: Due to the high computational cost of processing full articles with LLMs, the authors **limited the input to the abstracts** of articles when testing certain models, like GPT-4, Claude-3-Opus, and Gemini-1.5-Pro. This limitation was specifically mentioned for the test dataset. This introduces a gap between training and testing, and the results should be interpreted with this limitation in mind.

#### Models

The study evaluated multiple state-of-the-art large language models for generating lay summaries. These models were benchmarked against a **BART** baseline and included:

- **Claude-3-Opus** (Anthropic)
- **Gemini-1.5-Pro** (Google)
- **GPT-4** (OpenAI)
- **Llama3-8B** and **Llama3-70B** (Meta)

These models consist of billions to hundreds of billions of parameters, with advanced instruction-following and text-generation capabilities. The abstract and metadata of the articles were used as inputs due to token limitations and to reduce computational cost​.
##### Fine-Tuning for Relevance and Factuality

To improve relevance and factual consistency, the team applied **Low-Rank Adaptation (LoRA)** for fine-tuning. Fine-tuning was carried out separately for the **PLOS** and **eLife** datasets, as they have different characteristics in summary length and structure. An instruction-tuned prompt, "Write a lay summary suitable for non-experts," was used to guide the generation of summaries. Up to **8,000 tokens** from the full text were used as input for fine-tuning, and this process ran for **3 epochs** with a learning rate of **5e-5**​.

##### K-shot Prompting for Factuality

To enhance factual consistency, **K-shot prompting** was used. The **BGE-M3** embedding model was employed to find semantically similar articles from the training set. These similar articles were then included in the LLM prompts as examples, providing grounding and context for the model’s generation process. **K=1** was selected due to time constraints. This method resulted in notable improvements in **AlignScore** and **SummaC**, enhancing the factuality of the generated summaries​.

##### LLM Rewriting for Readability

To further improve the readability of summaries, the team implemented an **LLM-rewrite** strategy. First, a **BART** model was fine-tuned to generate initial summaries. These summaries were then rewritten using **OpenBioLLM-70B**, a biomedical-specialized LLM. This approach significantly improved the readability of the summaries, especially in the **LENS** metric, while maintaining factual consistency and relevance.

#### Training & Technology

The experiments were conducted on a server equipped with **8 NVIDIA RTX 4090Ti 48 GB GPUs**. The **LoRA fine-tuning** process was executed with a learning rate of **5e-5** over **3 epochs**, allowing efficient fine-tuning of the models for the PLOS and eLife datasets. For local models like **Llama-3-8B**, the models were downloaded from Huggingface, and the training was executed on these high-performance GPUs.

To manage costs associated with API-based models (e.g., GPT-4, Claude-3), the input was limited to the abstract section, and only essential experiments were conducted using these models.

------------------------------------

### Findings

- The **Llama-3-8B model** performed exceptionally well in **readability** during the **BioLaySumm 2024 shared task**, securing **first place** in that category. However, despite this strength, the model struggled with **relevance** and **factuality**, ultimately finishing in **29th place overall**. The trade-off between readability and other core metrics like factuality and relevance indicates that while the model generated summaries that were easy to understand, they sometimes lacked accuracy and precision.

- **Title infusing** played a critical role in enhancing the performance of models across the board. By incorporating the article title into the prompts, models were better positioned to generate more relevant and coherent summaries, as the title provided an essential context that helped guide the summarization process

- **Instruction tuning** using the article-summary pairs helped the Llama-3-8B model generate summaries that maintained a strong balance of **relevance** and **factuality**. The fine-tuned version of the model consistently outperformed its baseline version in both **relevance** (ROUGE-1, ROUGE-2, ROUGE-L) and **factuality** metrics (AlignScore and SummaC).
    
- The use of **K-shot prompting** allowed the model to leverage examples of semantically similar articles, which improved its **factuality** scores significantly. This technique demonstrated that grounding summaries in similar content from the training set aids in enhancing the accuracy of generated summaries, as evidenced by improvements in metrics like **AlignScore** and **SummaC**.

- **LLM rewriting**, where the initial summaries generated by a fine-tuned BART model were rewritten by a larger model like **OpenBioLLM-70B**, led to substantial improvements in **readability**. The rewritten summaries were easier for lay audiences to understand, achieving particularly high **LENS scores**, which measure alignment with human readability preferences. However, this improvement in readability came with only marginal gains in factual accuracy and relevance.

- Despite various improvements, the models continued to face difficulties in balancing **readability**, **relevance**, and **factuality**. Larger models, while excelling in readability, often struggled with factual consistency, while smaller models like **BART** maintained higher factual accuracy but fell short on readability.
    
- The decision to use only abstracts during testing led to a reduction in contextual information, which negatively impacted the relevance of the summaries to the original content. This limitation indicates the need for models to consider a more comprehensive input beyond just abstracts to improve overall summary quality.

------------------------------------

### Discussion

The BioLaySumm 2024 results underscore the complexities of using LLMs for biomedical summarization. **Llama-3-8B’s** strong **readability** performance demonstrates that LLMs are effective at producing accessible summaries, but the model’s **lower rankings in relevance and factuality** highlight a critical trade-off. In specialized domains like biomedicine, readability cannot come at the cost of **accuracy** and **relevance**. This balance is essential because lay summaries must be both easy to read and factually reliable, particularly when conveying medical information.

The impact of **title infusing** reveals the importance of contextual elements in guiding summarization. Titles provided a simple but effective way to anchor the models, ensuring that summaries better aligned with the article’s core focus. This suggests that future approaches could benefit from leveraging even more structural elements of the text, such as section headings, to further improve **relevance**.

**Instruction tuning** played a key role in helping the model maintain a better balance between **factuality and readability**, showing that targeted fine-tuning enhances a model’s ability to handle domain-specific tasks. However, the improvement was not sufficient to overcome the inherent challenges of summarizing complex biomedical literature. This points to the need for continued refinement of training methods to boost both accuracy and ease of understanding.

The success of **K-shot prompting** in improving **factuality** scores demonstrates that providing models with relevant examples enhances their ability to generate accurate summaries. This technique underscores the value of contextual grounding, suggesting that more sophisticated prompting strategies could further improve **factual consistency** without sacrificing readability.

The **LLM rewriting** strategy offers an interesting insight into the potential for hybrid approaches, where smaller, factually accurate models are combined with larger, more readable ones. While this method improved **readability**, the limited gains in **factuality and relevance** suggest that simply rewriting text does not fully address the accuracy issue. More integrated solutions are needed to ensure that improved readability doesn’t come at the expense of critical content.

Finally, the decision to test models on **abstract-only inputs** limited their ability to capture the full nuance of the research. Including more context would likely improve **relevance**, pointing to a broader need for models to work with richer inputs.

In conclusion, while **LLMs show great potential** for generating readable summaries, ensuring **factual accuracy and relevance** remains a challenge. Future work should focus on refining **contextual inputs** and exploring hybrid methods that balance these competing demands. Achieving this balance will be crucial for making LLMs truly effective in biomedical lay summarization.

------------------------------------

### Remarks & Limitations

- The **Llama-3-8B model** excelled in **readability**, but its trade-offs in **relevance** and **factuality** highlight the difficulty of balancing multiple metrics in biomedical summarization tasks, especially when optimizing for lay audiences.
    
- **Title infusing** improved the models' ability to generate relevant summaries, but the technique is dependent on accurate retrieval of titles, which may not always be available or relevant for all article types.
    
- **Instruction tuning** enhanced performance in **factuality** and **relevance**, but the improvements were constrained by the specific dataset used (eLife and PLOS). The results may not generalize to other types of biomedical documents.
    
- The **K-shot prompting** method significantly improved factual accuracy by grounding the model with similar examples, but it is computationally expensive and may not scale efficiently with larger datasets or real-time applications.
    
- The **LLM rewriting** strategy improved **readability** but yielded only marginal gains in **factuality** and **relevance**, suggesting that simply rewriting existing summaries may not be sufficient for addressing core challenges in accuracy.
    
- The use of **abstract-only inputs** in the test phase limited the models' ability to fully capture the nuances of the research, affecting the **relevance** of the summaries. Incorporating more comprehensive input data could lead to better performance across all metrics.
    
- The models were tested primarily on **biomedical research articles** from eLife and PLOS, focusing on life sciences, which may limit the generalizability of the findings to other scientific disciplines.
    
- **Computational limitations** restricted the extent of **fine-tuning and experimentation**. The models, particularly larger ones like **Llama-3-8B**, required significant resources, which could affect reproducibility and limit experimentation in resource-constrained settings.
    
- The closed-source nature of **OpenBioLLM-70B** and other proprietary models introduces challenges for **reproducibility** and future research, particularly as these models evolve without clear versioning.

- The ethical implications of using LLMs for generating lay summaries in highly sensitive biomedical fields are significant, especially given the risk of misinformation due to LLMs’ hallucination issues.

------------------------------------

### Citation

```
@inproceedings{zhou-etal-2024-team,
    title = "Team {YXZ} at {B}io{L}ay{S}umm: Adapting Large Language Models for Biomedical Lay Summarization",
    author = "Zhou, Jieli  and
      Ye, Cheng  and
      Xu, Pengcheng  and
      Xin, Hongyi",
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
    url = "https://aclanthology.org/2024.bionlp-1.76",
    pages = "818--825",
    abstract = "Biomedical literature are crucial for disseminating new scientific findings. However, the complexity of these research articles often leads to misinterpretations by the public. To address this urgent issue, we participated in the BioLaySumm task at the 2024 ACL BioNLP workshop, which focuses on automatically simplifying technical biomedical articles for non-technical audiences. We conduct a systematic evaluation of the SOTA large language models (LLMs) in 2024 and found that LLMs can generally achieve better readability scores than smaller models like Bart. Then we iteratively developed techniques of title infusing, K-shot prompting , LLM rewriting and instruction finetuning to further boost readability while balancing factuality and relevance. Notably, our submission achieved the first place in readability at the workshop, and among the top-3 teams with the highest readability scores, we have the best overall rank. Here, we present our experiments and findings on how to effectively adapt LLMs for automatic lay summarization. Our code is available at https://github.com/zhoujieli/biolaysumm.",
}
```