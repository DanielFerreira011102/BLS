# Datasets

This directory contains the datasets used throughout the project. The datasets are critical for the analysis and outcomes discussed within this repository.

## Accessing the Datasets

You can access the datasets through the following link:

[Download Dataset from Google Drive](https://drive.google.com/drive/folders/11DirHYh-yo7Q7wDjhUWCtRFg_1OQeizt?usp=sharing)

## Dataset Descriptions

The datasets used in this project are as follows:

| Dataset        | Question Type                                             | Answer Type      | Provides Context? |
|----------------|-----------------------------------------------------------|------------------|-------------------|
| MMLU-Med       | Which of the following best describes ...?                | A / B / C / D    | No                |
| MedQA          | A 72-year old man comes to the physician ...?             | A / B / C / D    | No                |
| MedMCQA        | Axonal transport is ...?                                  | A / B / C / D    | No                |
| PubMedQA       | Is anorectal endosonography valuable ...?                 | Yes / No / Maybe | Yes               |
| MEDIQA-AnS     | My father is suffering from IBS and is ...                | Summary          | Yes               |
| BioASQ-Y/N     | Is medical hydrology the same as Spa ...?                 | Yes / No         | Yes               |
| BioASQ-Factoid | Which enzyme is targeted by the drug ...?                 | Short answer     | Yes               |
| BioASQ-List    | Which proteins are involved in ...?                       | List             | Yes               |
| BioASQ-Summary | What is the effect of TRH on ...?                         | Summary          | Yes               |
| LiveQA         | What is the most common cause of ...?                     | Long answer      | No                |
| MedicationQA   | Does cyclosporine ophthalmic help ...?                    | Long answer      | No                |
| eLife          | \<Scientific research paper\>                             | Summary          | Yes               |
| PLOS           | \<Scientific research paper\>                             | Summary          | Yes               |
| HealthSearchQA | Are benign brain tumors serious?                          | n/a              | No                |

## Hugging Face Datasets

The datasets are also available on the Hugging Face Datasets library. You can access the datasets through the following links:
- [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- [MMLU](https://huggingface.co/datasets/cais/mmlu)
    - The MMUL dataset consists of multiple-choice questions from multiple branches of knowledge, including medicine, history, and physics. To get the medical questions, we need to access the links for the sub-datasets.
- [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)
- [LiveQA](https://huggingface.co/datasets/hyesunyun/liveqa_medical_trec2017)
- [MedicationQA](https://huggingface.co/datasets/truehealth/medicationqa)
- [MultiMedQA](https://huggingface.co/collections/openlifescienceai/multimedqa-66098a5b280539974cefe485)