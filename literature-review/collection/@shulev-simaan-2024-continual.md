---
title: Continual Reinforcement Learning for Controlled Text Generation
authors: Velizar Shulev, Khalil Sima’an
year: 2024
database: ACL Anthology
citekey: 10.1093-jamia-ocac149
tags:
  - reinforcement-learning
  - controlled-text-generation
  - ctg
url: https://aclanthology.org/2024.lrec-main.343/
file: "[[Continual Reinforcement Learning for Controlled Text Generation.pdf]]"
---

>[!title]
Continual Reinforcement Learning for Controlled Text Generation

>[!year]
2024

>[!author]
Velizar Shulev, Khalil Sima’an


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
@inproceedings{shulev-simaan-2024-continual,
    title = "Continual Reinforcement Learning for Controlled Text Generation",
    author = "Shulev, Velizar  and
      Sima{'}an, Khalil",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.343/",
    pages = "3881--3889",
    abstract = "Controlled Text Generation (CTG) steers the generation of continuations of a given context (prompt) by a Large Language Model (LLM) towards texts possessing a given attribute (e.g., topic, sentiment). In this paper we view CTG as a Continual Learning problem: how to learn at every step to steer next-word generation, without having to wait for end-of-sentence. This continual view is useful for online applications such as CTG for speech, where end-of-sentence is often uncertain. We depart from an existing model, the Plug-and-Play language models (PPLM), which perturbs the context at each step to better predict next-words that posses the desired attribute. While PPLM is intricate and has many hyper-parameters, we provide a proof that the PPLM objective function can be reduced to a Continual Reinforcement Learning (CRL) reward function, thereby simplifying PPLM and endowing it with a better understood learning framework. Subsequently, we present, the first of its kind, CTG algorithm that is fully based on CRL and exhibit promising empirical results."
}
```