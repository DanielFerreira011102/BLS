---
title: Capabilities of Gemini Models in Medicine
authors: Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, Juanma Zambrano Chaves, Szu-Yeu Hu, Mike Schaekermann, Aishwarya Kamath, Yong Cheng, David G.T. Barrett, Cathy Cheung, Basil Mustafa, Anil Palepu, Daniel McDuff, Le Hou, Tomer Golany, Luyang Liu, Jean-baptiste Alayrac, Neil Houlsby, Nenad Tomasev, Jan Freyberg, Charles Lau, Jonas Kemp, Jeremy Lai, Shekoofeh Azizi, Kimberly Kanada, SiWai Man, Kavita Kulkarni, Ruoxi Sun, Siamak Shakeri, Luheng He, Ben Caine, Albert Webson, Natasha Latysheva, Melvin Johnson, Philip Mansfield, Jian Lu, Ehud Rivlin, Jesper Anderson, Bradley Green, Renee Wong, Jonathan Krause, Jonathon Shlens, Ewa Dominowska, S. M. Ali Eslami, Katherine Chou, Claire Cui, Oriol Vinyals, Koray Kavukcuoglu, James Manyika, Jeff Dean, Demis Hassabis, Yossi Matias, Dale Webster, Joelle Barral, Greg Corrado, Christopher Semturs, S. Sara Mahdavi, Juraj Gottweis, Alan Karthikesalingam, Vivek Natarajan
year: 2024
database: arXiv
citekey: saab2024capabilitiesgeminimodelsmedicine
tags:
  - med-gemini
  - gemini/1/0/pro
  - usmle
  - medqa-us
  - medqa-rs
  - med-gemini-s/1/0
  - gemini/m/1/5
  - med-gemini-m/1/5
  - gemini/1/5/pro
  - ehr
  - self-training
  - cot
url: https://arxiv.org/abs/2404.18416
file: "[[Capabilities of Gemini Models in Medicine.pdf]]"
---

>[!title]
Capabilities of Gemini Models in Medicine

>[!year]
2024

>[!author]
Khaled Saab, Tao Tu, Wei-Hung Weng, Ryutaro Tanno, David Stutz, Ellery Wulczyn, Fan Zhang, Tim Strother, Chunjong Park, Elahe Vedadi, Juanma Zambrano Chaves, Szu-Yeu Hu, Mike Schaekermann, Aishwarya Kamath, Yong Cheng, David G.T. Barrett, Cathy Cheung, Basil Mustafa, Anil Palepu, Daniel McDuff, Le Hou, Tomer Golany, Luyang Liu, Jean-baptiste Alayrac, Neil Houlsby, Nenad Tomasev, Jan Freyberg, Charles Lau, Jonas Kemp, Jeremy Lai, Shekoofeh Azizi, Kimberly Kanada, SiWai Man, Kavita Kulkarni, Ruoxi Sun, Siamak Shakeri, Luheng He, Ben Caine, Albert Webson, Natasha Latysheva, Melvin Johnson, Philip Mansfield, Jian Lu, Ehud Rivlin, Jesper Anderson, Bradley Green, Renee Wong, Jonathan Krause, Jonathon Shlens, Ewa Dominowska, S. M. Ali Eslami, Katherine Chou, Claire Cui, Oriol Vinyals, Koray Kavukcuoglu, James Manyika, Jeff Dean, Demis Hassabis, Yossi Matias, Dale Webster, Joelle Barral, Greg Corrado, Christopher Semturs, S. Sara Mahdavi, Juraj Gottweis, Alan Karthikesalingam, Vivek Natarajan


------------------------------------

### Summary

- The paper introduces **Med-Gemini**, a family of advanced multimodal models specialized in medicine. It builds on the general Gemini models with enhancements for **clinical reasoning**, **web search integration**, and handling **multimodal data** (like text, images, videos, and health records).
    
- Med-Gemini is designed to perform **complex clinical reasoning** and can **improve accuracy** by using web searches during inference, especially when encountering **uncertainty in diagnoses**.
    
- It employs a **novel uncertainty-guided search strategy** to enhance accuracy in complex reasoning tasks, integrating **web search** to stay current with the latest medical knowledge.
    
- The models are tailored to handle a wide variety of **data types**, including **long electronic health records (EHRs)**, medical images, videos, and text, enabling them to excel in **diagnostic** and **medical summarization** tasks.
    
- Med-Gemini establishes **state-of-the-art (SoTA) performance** on 10 out of 14 medical benchmarks, including an impressive **91.1% accuracy on MedQA (USMLE)**, significantly surpassing **GPT-4** and previous models (e.g., Med-PaLM 2).
    
- The model also performed strongly on complex diagnostic cases from the **New England Journal of Medicine (CPC cases)** and **GeneTuring benchmarks**.
    
- Med-Gemini outperformed **GPT-4V** on **multimodal medical tasks**, such as **visual question-answering**, with an average margin of **44.5%**.
    
- The model can be **fine-tuned** for specific medical tasks using **custom encoders**, such as for interpreting **electrocardiograms (ECGs)** and other medical signals.
    
- Beyond benchmarks, Med-Gemini demonstrated real-world potential in tasks like generating **medical summaries**, **referral letters**, and **simplifying medical language** for patient communication. It even **outperformed human experts** in some tasks like medical note summarization.
    
- The paper showcases Med-Gemini's ability to handle **multimodal medical dialogues**, with examples in **dermatology** and **radiology**, where the model interacts based on images and medical information.
    
- Despite its impressive results, the paper emphasizes that **rigorous evaluation** is needed before deploying Med-Gemini in real-world medical settings due to the **critical nature of medical decision-making**.



------------------------------------

### Research question

How can a specialized multimodal AI model like Med-Gemini be developed to handle complex clinical reasoning, multimodal data, and long-context understanding to surpass existing models (like GPT-4) in medical tasks and benchmarks, while ensuring real-world applicability in medicine?

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
@misc{saab2024capabilitiesgeminimodelsmedicine,
      title={Capabilities of Gemini Models in Medicine}, 
      author={Khaled Saab and Tao Tu and Wei-Hung Weng and Ryutaro Tanno and David Stutz and Ellery Wulczyn and Fan Zhang and Tim Strother and Chunjong Park and Elahe Vedadi and Juanma Zambrano Chaves and Szu-Yeu Hu and Mike Schaekermann and Aishwarya Kamath and Yong Cheng and David G. T. Barrett and Cathy Cheung and Basil Mustafa and Anil Palepu and Daniel McDuff and Le Hou and Tomer Golany and Luyang Liu and Jean-baptiste Alayrac and Neil Houlsby and Nenad Tomasev and Jan Freyberg and Charles Lau and Jonas Kemp and Jeremy Lai and Shekoofeh Azizi and Kimberly Kanada and SiWai Man and Kavita Kulkarni and Ruoxi Sun and Siamak Shakeri and Luheng He and Ben Caine and Albert Webson and Natasha Latysheva and Melvin Johnson and Philip Mansfield and Jian Lu and Ehud Rivlin and Jesper Anderson and Bradley Green and Renee Wong and Jonathan Krause and Jonathon Shlens and Ewa Dominowska and S. M. Ali Eslami and Katherine Chou and Claire Cui and Oriol Vinyals and Koray Kavukcuoglu and James Manyika and Jeff Dean and Demis Hassabis and Yossi Matias and Dale Webster and Joelle Barral and Greg Corrado and Christopher Semturs and S. Sara Mahdavi and Juraj Gottweis and Alan Karthikesalingam and Vivek Natarajan},
      year={2024},
      eprint={2404.18416},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2404.18416}, 
}
```