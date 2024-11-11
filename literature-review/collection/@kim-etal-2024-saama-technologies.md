---
title: "Saama Technologies at BioLaySumm: Abstract based fine-tuned models with LoRA"
authors: Hwanmun Kim, Kamal raj Kanakarajan, Malaikannan Sankarasubbu
year: 2024
database: ACL Anthology
citekey: kim-etal-2024-saama-technologies
tags:
  - biolaysumm/2024
  - plos
  - elife
  - rouge
  - bert-score
  - fkgl
  - dcrs
  - cli
  - lens
  - align-score
  - summa-c
  - lora
  - mistral/7b/instruct/0/2
  - v-llm
  - transformers
  - peft
  - trl
  - dpo
  - zero-shot
  - few-shot
url: https://aclanthology.org/2024.bionlp-1.72/
file: "[[Saama Technologies at BioLaySumm - Abstract based fine-tuned models.pdf]]"
---

>[!title]
Saama Technologies at BioLaySumm: Abstract based fine-tuned models with LoRA

>[!year]
2024

>[!author]
Hwanmun Kim, Kamal raj Kanakarajan, Malaikannan Sankarasubbu

------------------------------------

### Summary

- This paper details Saama Technologies' participation in the **BioLaySumm 2024** challenge, focusing on the **lay summarization of biomedical research articles**. 

- The aim is to make complex biomedical information accessible to a broader audience.

- The authors explored various **fine-tuning strategies**, utilizing the **LoRA (Low-Rank Adaptation)** technique to enhance summarization quality based on abstracts from the provided research articles.

- Their best-performing model was an **unsupervised fine-tuned model with LoRA**, which was further refined through a post-processing step to eliminate repetitive sentences, achieving a rank of **3rd overall** in the BioLaySumm 2024 leaderboard.

- The challenge involved generating lay summaries from the **PLOS** and **eLife** datasets, which were annotated with ground-truth lay summaries for validation.

- The evaluation of summaries was based on three key criteria: **relevance**, **readability**, and **factuality**, employing metrics like **ROUGE**, **BERTScore**, and various readability scores.

- The paper discusses several experimental approaches, including **zero-shot** and **few-shot prompting**, **supervised fine-tuning**, and **direct preference optimization (DPO)**, to improve model performance.

- Results indicated that while their model excelled in relevance, it faced challenges in readability and factuality, suggesting a trade-off between these criteria.

- The authors highlight limitations due to their use of a single type of smaller open-source model and express interest in future work that could utilize larger models and address the full content of research articles for improved summarization.

------------------------------------

### Research question

 How can large language models (LLMs) be adapted and fine-tuned to generate accurate, relevant, and readable lay summaries of biomedical research articles for non-expert readers?

------------------------------------

### Context

The paper was created in the context of the 23rd Workshop on Biomedical Language Processing (BioNLP 2024), which included a shared task called BioLaySumm. The task addresses the growing challenge of making biomedical research accessible to laypeople. With the rapid increase in scientific publications, particularly in the biomedical field, lay audiences struggle to keep up due to the technical complexity of these studies. Therefore, lay summarization—a form of text summarization that focuses on making content more understandable for non-experts—is needed.

Although summarization of scientific papers has been widely studied, creating summaries specifically tailored to non-expert audiences is a relatively new and critical area of focus, especially given the importance of biomedical research in improving public health.

------------------------------------

### Methodology

#### Datasets

The models were trained and evaluated using the **PLOS** and **eLife** datasets, both of which are common in biomedical research and include full-text articles as well as their lay summaries. The **PLOS** dataset contains 24,773 articles in the training set and 1,376 in the validation set, while the **eLife** dataset comprises 4,346 training articles and 241 validation articles. Each article in these datasets comes with its respective abstract, keywords, and, for the training and validation splits, ground-truth lay summaries provided either by the authors (PLOS) or expert editors (eLife). The test set for each dataset includes 142 articles, and the task focuses on generating accurate and readable lay summaries for non-expert audiences.

#### Evaluation Metrics

The evaluation of the summarization models was based on three core criteria: **Relevance**, **Readability**, and **Factuality**, using the following metrics:

- **Relevance**: Measured using **ROUGE** (1, 2, and L) and **BERTScore**.
- **Readability**: Assessed through **Flesch-Kincaid Grade Level (FKGL)**, **Dale-Chall Readability Score (DCRS)**, **Coleman-Liau Index (CLI)**, and **LENS** (a learnable evaluation metric for simplification).
- **Factuality**: Evaluated using **AlignScore** and **SummaC** to ensure consistency and accuracy.

The goal of the model was to maximize scores in ROUGE and BERTScore while minimizing FKGL, DCRS, and CLI to ensure both relevance and readability, along with maintaining factual accuracy.

#### Preprocessing

Because full research articles are often too long to fit into the model's input, the fine-tuning process used only the **abstract** of the article as input. The abstract gives a concise summary of the article’s main points, making it an ideal input for generating a lay summary. This approach avoids overloading the model with unnecessary information and focuses on the key content.

#### Model 1: Unsupervised Fine-tuned LoRA Model

The primary model used was an **unsupervised fine-tuned model** leveraging **LoRA** (Low-Rank Adaptation), a technique that enables more efficient fine-tuning of large models by reducing the number of trainable parameters.

- **Unsupervised fine-tuning** means the model was trained using both the abstract and the corresponding lay summary without explicitly labeling which part of the input was the abstract and which part was the summary. Instead, the model learned to generate lay summaries by identifying patterns between the abstracts and summaries in the training data.
- For text generation, the model was fed just the **abstract** of the article during test time and asked to generate a new lay summary.

This unsupervised approach allowed the model to learn generalized relationships between abstracts and lay summaries across the dataset, without needing a structured format for input-output pairs during training.
#### Model 2: Supervised Fine-tuning with LoRA

The second model employed **supervised fine-tuning** with **LoRA**. In this approach, the model was explicitly trained with a clear distinction between the **input** (the abstract) and the **output** (the lay summary), providing more structured guidance for learning.

- The **abstract** of each article was treated as the input text, while the **lay summary** was treated as the target output that the model needed to generate.
- The model was trained specifically to produce lay summaries by learning to simplify the abstract for non-expert readers, ensuring that it generated summaries based on the core ideas from the abstract without simply mimicking its structure or language.

This **supervised approach** provided clearer signals to the model, guiding it to understand that its task was to rephrase and simplify the abstract content, focusing on accessibility and clarity for general audiences.

#### Model 3: Direct Preference Optimization (DPO)

To further improve the **factual accuracy** of the generated summaries, the authors employed **Direct Preference Optimization (DPO)**. This technique refines the model's outputs by preferring summaries that are more factually accurate.

- The model was used to generate lay summaries for **1,000 randomly selected articles** from the training dataset.
- The factual consistency of each generated summary was compared against the ground-truth lay summaries using **AlignScore** and **SummaC**, two metrics designed to evaluate the factuality of generated text.
- Summaries were then ranked based on these factuality scores, and the model was fine-tuned to prefer summaries that were more factually accurate according to these rankings.

By applying DPO, the model became more adept at generating summaries that not only read well but were also factually aligned with the original research, reducing the risk of generating misleading or incorrect information.

#### Postprocessing

After the model generated the lay summaries, a **postprocessing** step was applied to remove repeated sentences, which were a common issue in the raw output.

- The system split the generated text into individual sentences and checked for duplicates.
- If any sentence appeared more than once, the duplicates were removed to make the summary more concise.
- Special rules were implemented to avoid incorrect sentence splitting, such as ignoring periods within URLs, abbreviations (e.g., "e.g."), or numerical values.

This cleanup process helped ensure that the final summaries were concise and free of redundant information.

#### Training and Technology

All experiments were conducted on a system with **Quadra RTX 8000 GPUs**, each with 48 GB of memory, which provided the necessary computational power for training the models.

- The text generation was handled using the **Mistral-7B-instruct-v0.2** model.
- For fine-tuning, libraries from **Huggingface** were used, along with the **AdamW optimizer** and **cross-entropy loss** for training.
- The model was trained with a **sequence length of up to 4,096 tokens**, a **batch size of 8**, and **learning rates ranging from 1×10−5 to 2×10−5**.
- A **linear learning rate scheduler** was applied during the training process.
- **LoRA parameters** were also applied, with **r = 8** and **α = 16**.

The training was carried out over **3 to 5 epochs**, with parameters specifically tuned for the task of lay summarization.

------------------------------------

### Findings

- The unsupervised fine-tuned model with **LoRA** ranked 3rd overall in the BioLaySumm 2024 leaderboard, showing strong relevance scores. This suggests that focusing solely on the abstract portion of biomedical articles and employing low-rank adaptation (LoRA) can effectively enhance the relevance of lay summaries.
    
- The model scored particularly high in **relevance**, outperforming in ROUGE and BERTScore metrics and achieving 2nd place in the competition. However, **readability** and **factuality** were lower, with the model ranking 16th and 18th respectively, indicating challenges in simplifying language and maintaining fact-checking rigor.

- The **Supervised Fine-tuning approach** did not outperform the unsupervised model overall. It showed a slight advantage in **readability** scores in certain cases, but generally fell behind in relevance and factuality. It also did not deliver the expected gains in simplifying abstract information for lay audiences.
    
- **Direct Preference Optimization (DPO)** improved **factuality** and **readability** of the generated summaries. However, DPO caused a noticeable reduction in **relevance** scores, suggesting that the fine-tuned model struggled to balance factual accuracy with the preservation of key summary content.
    
- **Few-shot prompting** provided decent results as a baseline but underperformed compared to the fine-tuning approaches. It was relatively weaker in maintaining factuality and readability across datasets, but more consistent in relevance.
    
- **Post-processing** had a mixed impact. On one hand, it enhanced **factuality** and **relevance**, successfully eliminating repeated sentences from summaries. On the other hand, post-processing had inconsistent effects on **readability**, sometimes improving metrics like LENS but negatively impacting DCRS and CLI scores.
    
- There was a **trade-off between readability and factuality**. For example, PLOS summaries showed poorer readability but better factuality, while the opposite was true for eLife datasets, indicating that simplified summaries tended to lose accuracy in conveying complex information.
    
- The **abstract-based input** approach limited the model's exposure to the entire research article, resulting in summaries that sometimes failed to capture the full factual detail of the original paper, especially for highly complex or long articles.

------------------------------------

### Discussion

The experiments in this study underscore the complexities involved in generating high-quality lay summaries for biomedical research articles. The **unsupervised fine-tuned LoRA model** performed particularly well in relevance, reflecting that fine-tuning on abstracts is a viable strategy for condensing dense technical content into more digestible summaries. However, the lower scores in factuality for certain datasets, such as eLife, suggest that relying solely on abstract-based fine-tuning might lead to incomplete or inaccurate representations of the full article. This indicates that, while abstracts provide a useful starting point, future models should explore ways to incorporate more content from the body of the articles.

The **supervised fine-tuning** approach showed limited gains in readability, but it did not translate to significantly better overall performance, especially in terms of relevance and factuality. This may be due to the supervised model's inability to fully disentangle the language patterns of abstracts from those of lay summaries, leading to summaries that remained overly technical or failed to simplify key concepts effectively for non-expert readers.

The application of **Direct Preference Optimization** introduced a clear improvement in factual consistency but at the cost of relevance. This highlights a key trade-off in text summarization: factuality often requires the inclusion of detailed, precise information, but such details can detract from the readability and overall flow of a summary designed for lay audiences. Moving forward, a more balanced approach that integrates DPO into the fine-tuning process without sacrificing relevance will be crucial. This could involve leveraging larger datasets or more sophisticated techniques for dynamically prioritizing both accuracy and content flow.

**Post-processing**, while effective in mitigating sentence repetition, had an unpredictable effect on readability. This suggests that automatic post-processing mechanisms might need further refinement to ensure that readability improvements are consistent across different datasets. Additionally, the mixed outcomes on metrics like DCRS and CLI point to a broader challenge in ensuring that the linguistic complexity of summaries remains appropriate for non-expert audiences without compromising the integrity of the information presented.

The contrasting performance between PLOS and eLife datasets also speaks to the **trade-offs between readability and factuality**. The lower readability but higher factuality of PLOS summaries might reflect the fact that lay summaries authored by researchers tend to retain more technical language. Conversely, the more readable but less factually rigorous eLife summaries point to potential over-simplifications. Addressing this balance will be key for future research, as the goal is to make the content accessible without diluting its scientific validity.

Finally, the reliance on **abstract-based inputs** in our models limited their ability to fully capture the complexities of the original research articles. Abstracts, while concise, often leave out nuanced data or detailed findings crucial for factual accuracy. Future work should explore the use of models capable of handling larger context windows to better process entire research articles. This could improve the overall factuality and depth of the generated summaries, helping to ensure they remain both informative and accessible.

In summary, the experiments revealed that achieving the right balance between relevance, readability, and factuality is an ongoing challenge in biomedical lay summarization. While the LoRA-based fine-tuning approach shows promise, future work will need to refine techniques such as DPO and post-processing to achieve a better integration of these three key aspects. The exploration of hybrid approaches that combine extractive and abstractive methods may offer additional pathways to optimizing performance across diverse biomedical content.

------------------------------------

### Remarks & Limitations

- The **limited resources** available restricted the experiments to a single, relatively small, open-source model. As a result, larger, more advanced models could not be explored, potentially impacting overall performance.

- **Context size limitations** of the selected model constrained its ability to process full research articles, which could lead to incomplete or factually inconsistent summaries. This limitation was partly mitigated by using **Direct Preference Optimization (DPO)**, but DPO's interaction with full articles was restricted to factuality scores rather than full content analysis.

- The approach was particularly **successful in relevance** but underperformed in readability and factuality. This imbalance suggests that while the model effectively captured important content, its ability to simplify information and maintain factual integrity was less robust.

- **Readability challenges** may have stemmed from the fact that summaries more readable than the golden standard tended to score lower on BERTScore. This suggests a potential bias in evaluation methods, where simplified, lay-friendly summaries were penalized in terms of relevance or factuality.

- The experiments did not include benchmarking across different summary evaluation criteria, which could have provided a broader understanding of model performance beyond the current competition metrics.

- Future research should focus on exploring **larger models** or those with better **contextual capacity**, along with more diverse evaluation criteria, to enhance both factuality and readability without sacrificing relevance.

------------------------------------

### Citation

```
@inproceedings{kim-etal-2024-saama-technologies,
    title = "Saama Technologies at {B}io{L}ay{S}umm: Abstract based fine-tuned models with {L}o{RA}",
    author = "Kim, Hwanmun  and
      Kanakarajan, Kamal raj  and
      Sankarasubbu, Malaikannan",
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
    url = "https://aclanthology.org/2024.bionlp-1.72",
    pages = "786--792",
    abstract = "Lay summarization of biomedical research articles is a challenging problem due to their use of technical terms and background knowledge requirements, despite the potential benefits of these research articles to the public. We worked on this problem as participating in BioLaySumm 2024. We experimented with various fine-tuning approaches to generate better lay summaries for biomedical research articles. After several experiments, we built a LoRA model with unsupervised fine-tuning based on the abstracts of the given articles, followed by a post-processing unit to take off repeated sentences. Our model was ranked 3rd overall in the BioLaySumm 2024 leaderboard. We analyzed the different approaches we experimented with and suggested several ideas to improve our model further.",
}
```