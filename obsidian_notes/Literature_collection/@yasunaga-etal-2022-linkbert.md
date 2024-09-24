---
title: "LinkBERT: Pretraining Language Models with Document Links"
authors: Michihiro Yasunaga, Jure Leskovec, Percy Liang
year: 2022
database: ACL Anthology
citekey: yasunaga-etal-2022-linkbert
tags:
  - BERT
url: https://aclanthology.org/2022.acl-long.551/
file: "[[LinkBERT - Pretraining Language Models with Document Links.pdf]]"
---

>[!title]
LinkBERT: Pretraining Language Models with Document Links

>[!year]
2022

>[!author]
Michihiro Yasunaga, Jure Leskovec, Percy Liang


------------------------------------

### Summary
1. 

------------------------------------

### Research question
1. 

------------------------------------

### Findings
1. 

------------------------------------

### Discussion
1. 

------------------------------------

### Methodology
1. 

------------------------------------

### Remarks
1. 

------------------------------------

### Citation

```
@inproceedings{yasunaga-etal-2022-linkbert,
    title = "{L}ink{BERT}: Pretraining Language Models with Document Links",
    author = "Yasunaga, Michihiro  and
      Leskovec, Jure  and
      Liang, Percy",
    editor = "Muresan, Smaranda  and
      Nakov, Preslav  and
      Villavicencio, Aline",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.551",
    doi = "10.18653/v1/2022.acl-long.551",
    pages = "8003--8016",
    abstract = "Language model (LM) pretraining captures various knowledge from text corpora, helping downstream tasks. However, existing methods such as BERT model a single document, and do not capture dependencies or knowledge that span across documents. In this work, we propose LinkBERT, an LM pretraining method that leverages links between documents, e.g., hyperlinks. Given a text corpus, we view it as a graph of documents and create LM inputs by placing linked documents in the same context. We then pretrain the LM with two joint self-supervised objectives: masked language modeling and our new proposal, document relation prediction. We show that LinkBERT outperforms BERT on various downstream tasks across two domains: the general domain (pretrained on Wikipedia with hyperlinks) and biomedical domain (pretrained on PubMed with citation links). LinkBERT is especially effective for multi-hop reasoning and few-shot QA (+5{\%} absolute improvement on HotpotQA and TriviaQA), and our biomedical LinkBERT sets new states of the art on various BioNLP tasks (+7{\%} on BioASQ and USMLE). We release our pretrained models, LinkBERT and BioLinkBERT, as well as code and data.",
}
```