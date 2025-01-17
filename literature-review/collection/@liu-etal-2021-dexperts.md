---
title: "DEXPERTS: Decoding-Time Controlled Text Generation\rwith Experts and Anti-Experts"
authors: Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, Yejin Choi
year: 2021
database: ACL Anthology
citekey: liu-etal-2021-dexperts
tags:
  - ctg
  - controlled-text-generation
  - decoding-time-intervention
url: https://aclanthology.org/2021.acl-long.522/
file: "[[DEXPERTS - Decoding-time controlled text generation with experts and anti-experts.pdf]]"
---

>[!title]
>DEXPERTS: Decoding-Time Controlled Text Generation with Experts and Anti-Experts


>[!year]
2021

>[!author]
Alisa Liu, Maarten Sap, Ximing Lu, Swabha Swayamdipta, Chandra Bhagavatula, Noah A. Smith, Yejin Choi


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
@inproceedings{liu-etal-2021-dexperts,
    title = "{DE}xperts: Decoding-Time Controlled Text Generation with Experts and Anti-Experts",
    author = "Liu, Alisa  and
      Sap, Maarten  and
      Lu, Ximing  and
      Swayamdipta, Swabha  and
      Bhagavatula, Chandra  and
      Smith, Noah A.  and
      Choi, Yejin",
    editor = "Zong, Chengqing  and
      Xia, Fei  and
      Li, Wenjie  and
      Navigli, Roberto",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.522/",
    doi = "10.18653/v1/2021.acl-long.522",
    pages = "6691--6706",
    abstract = "Despite recent advances in natural language generation, it remains challenging to control attributes of generated text. We propose DExperts: Decoding-time Experts, a decoding-time method for controlled text generation that combines a pretrained language model with {\textquotedblleft}expert{\textquotedblright} LMs and/or {\textquotedblleft}anti-expert{\textquotedblright} LMs in a product of experts. Intuitively, under the ensemble, tokens only get high probability if they are considered likely by the experts, and unlikely by the anti-experts. We apply DExperts to language detoxification and sentiment-controlled generation, where we outperform existing controllable generation methods on both automatic and human evaluations. Moreover, because DExperts operates only on the output of the pretrained LM, it is effective with (anti-)experts of smaller size, including when operating on GPT-3. Our work highlights the promise of tuning small LMs on text with (un)desirable attributes for efficient decoding-time steering."
}
```