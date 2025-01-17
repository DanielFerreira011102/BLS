---
title: Critic-Guided Decoding for Controlled Text Generation
authors: Minbeom Kim, Hwanhee Lee, Kang Min Yoo, Joonsuk Park, Hwaran Lee, Kyomin Jung
year: 2023
database: ACL Anthology
citekey: kim-etal-2023-critic
tags:
  - controlled-text-generation
  - ctg
  - decoding-time-intervention
url: https://aclanthology.org/2023.findings-acl.281/
file: "[[Critic-Guided Decoding for Controlled Text Generation.pdf]]"
---

>[!title]
>Automated Text Simplification: A Survey


>[!year]
2023

>[!author]
Minbeom Kim, Hwanhee Lee, Kang Min Yoo, Joonsuk Park, Hwaran Lee, Kyomin Jung

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
@inproceedings{kim-etal-2023-critic,
    title = "Critic-Guided Decoding for Controlled Text Generation",
    author = "Kim, Minbeom  and
      Lee, Hwanhee  and
      Yoo, Kang Min  and
      Park, Joonsuk  and
      Lee, Hwaran  and
      Jung, Kyomin",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.281/",
    doi = "10.18653/v1/2023.findings-acl.281",
    pages = "4598--4612",
    abstract = "Steering language generation towards objectives or away from undesired content has been a long-standing goal in utilizing language models (LM). Recent work has demonstrated reinforcement learning and weighted decoding as effective approaches to achieve a higher level of language control and quality with pros and cons. In this work, we propose a novel critic decoding method for controlled language generation (CriticControl) that combines the strengths of reinforcement learning and weighted decoding. Specifically, we adopt the actor-critic framework and train an LM-steering critic from reward models. Similar to weighted decoding, our method freezes the language model and manipulates the output token distribution using a critic to improve training efficiency and stability. Evaluation of our method on three controlled generation tasks, topic control, sentiment control, and detoxification, shows that our approach generates more coherent and well-controlled texts than previous methods. In addition, CriticControl demonstrates superior generalization ability in zero-shot settings. Human evaluation studies also corroborate our findings."
}
```