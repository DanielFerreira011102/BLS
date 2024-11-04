---
title: "Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model"
authors: Haikang Deng, Colin Raffel
year: 2023
database: ACL Anthology
citekey: deng-raffel-2023-reward
tags:
  - CTRL
url: https://aclanthology.org/2023.emnlp-main.721/
file: "[[Reward-Augmented Decoding - Efficient Controlled Text Generation With a Unidirectional Reward Model.pdf]]"
---

>[!title]
Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model

>[!year]
2023

>[!author]
Haikang Deng, Colin Raffel


------------------------------------

### Summary


------------------------------------

### Research question


------------------------------------

### Context


------------------------------------

### Methodology


------------------------------------

### Findings


------------------------------------

### Discussion


------------------------------------

### Remarks & Limitations


------------------------------------

### Citation

```
@inproceedings{deng-raffel-2023-reward,
    title = "Reward-Augmented Decoding: Efficient Controlled Text Generation With a Unidirectional Reward Model",
    author = "Deng, Haikang  and
      Raffel, Colin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.721",
    doi = "10.18653/v1/2023.emnlp-main.721",
    pages = "11781--11791",
    abstract = "While large language models have proven effective in a huge range of downstream applications, they often generate text that is problematic or lacks a desired attribute. In this paper, we introduce Reward-Augmented Decoding (RAD), a text generation procedure that uses a small unidirectional reward model to encourage a language model to generate text that has certain properties. Specifically, RAD uses the reward model to score generations as they are produced and rescales sampling probabilities to favor high-reward tokens. By using a unidirectional reward model, RAD can cache activations from prior generation steps to decrease computational overhead. Through experiments on generating non-toxic and sentiment-controlled text, we demonstrate that RAD performs best among methods that change only the generation procedure and matches the performance of state-of-the-art methods that involve re-training the language model. We further validate that RAD is effective on very large language models while incurring a minimal computational overhead.",
}
```