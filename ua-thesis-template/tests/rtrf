# CTRL

\begin{equation}
\label{eq:ctrl-objective}
\mathcal{L}{CTRL} = -\sum{i=1}^{n} \log P(x_i | x_{<i}, c; \theta)
\end{equation}
where $x_i$ is the $i$-th token in the input sequence, $x_{<i}$ represents the tokens before position $i$, and $\theta$ denotes the model parameters.

\\\

In the context of medical question answering, retraining a language model from scratch would involve curating a large-scale medical corpus with annotations for different levels of text complexity. 
For example, medical articles could be labeled as "simple", "intermediate", or "advanced" based on their technicality and assumed reader background. 
The model would then learn to associate these complexity labels with the corresponding language patterns during training. 
However, creating such a large annotated medical dataset can be challenging and time-consuming, requiring input from domain experts.
Another consideration specific to medical language models is the importance of factual accuracy and avoidance of harmful misinformation. 
Retraining a model on medical data without proper filtering and validation could lead to the model learning and propagating inaccurate or misleading information. 
Therefore, careful curation and fact-checking of the training data is crucial when applying retraining approaches to medical text generation.

\\\

For our medical QA application, we would need a large corpus of medical text annotated with codes for complexity level, target audience, and potentially other attributes like specialty area. Obtaining such a dataset would be a significant challenge.

\\\

In the context of medical QA, retraining a language model with medical-specific control codes could potentially enable fine-grained control over the complexity and style of generated responses. 
However, the lack of large-scale labeled medical datasets and the computational demands of retraining pose significant challenges. 
Additionally, ensuring the factual accuracy and safety of the generated content would be critical in the medical domain.

\\\

The concept of control codes introduced by CTRL embodies the core intuition behind \gls{ctg} tasks and has laid a critical foundation for both retraining methods and the entire \gls{ctg} field. 
The retraining approach showcases considerable diversity in innovations related to training data, model architecture, and training methods \cite{keskar2019ctrlconditionaltransformerlanguage, lu2022quarkcontrollabletext, chan2021coconselfsupervisedapproach}. 
In the application of these methods, different control tasks, such as abstract attribute control tasks and concrete content control tasks, often exhibit distinct common characteristics.

\\\

At inference time, the desired control code is provided as part of the input, guiding the model to generate text with the specified attributes. 
While CTRL demonstrates the effectiveness of retraining for controlled generation, its applicability to the medical domain may be limited due to the lack of a large-scale medical corpus with explicit control codes.

\\\

At inference time, the control code acts as a knob to steer the generation:
\vspace{-0.3em}
\begin{lstlisting}[style=python, numbers=none]
[Science][Title] Researchers have discovered bacteria that thrive in high-CO2 environments.
[Horror][Text] When I was a little girl, my parents divorced. My dad left, and my mom took care of me.
[Reviews][Rating: 5.0] I've used this hair product for years. It keeps my hair soft without feeling greasy.
\end{lstlisting}
While conceptually simple, training a high-quality conditional LM like CTRL requires massive amounts of diverse data and compute resources. 
Collecting datasets with explicit control codes for every desired attribute is often infeasible, especially for specialized domains like medicine. 
The approach also lacks flexibility, as generating text with new attributes not seen during training would require retraining the entire model. 
Nevertheless, CTRL demonstrates the potential of conditional language modeling and has inspired various subsequent works.

\\\

However, training a model like CTRL from scratch requires substantial computational resources and a large corpus of domain-specific text, which may be challenging to curate for specialized domains like medicine.

\\\

However, while effective for broad attributes, CTRL may struggle with more nuanced control, such as fine-grained sentiment or complexity levels.

# POINTER

Another notable architectural modification approach is the Progressive Insertion-based Transformer (POINTER) model \cite{he2021parallelrefine}. POINTER enables lexically constrained text generation by modifying the Transformer architecture to generate text in a progressive manner. Given a set of constrained words, POINTER first generates these words to satisfy the lexical constraints and then iteratively inserts additional words between them until a complete sentence is formed.
The generation process in POINTER consists of two stages: token-level planning and iterative refinement. The planning stage employs a token-level classifier to predict the positions of the constrained words in the output sequence, followed by the iterative refinement stage where the model inserts words between the constrained tokens in parallel until the sentence is complete. The training objective for POINTER minimizes the cross-entropy loss for both the token-level classifier ($\mathcal{L}{cls}$) and the generation model ($\mathcal{L}{gen}$):
\begin{equation}
\label{eq:pointer-objective}
\mathcal{L}{POINTER} = \mathcal{L}{cls} + \mathcal{L}_{gen}
\end{equation}
While POINTER achieves high-quality lexically constrained generation by explicitly modeling the planning and refinement process, it assumes the availability of constrained words as input, which may not always be the case in practical applications. Additionally, the iterative refinement process can be computationally expensive, particularly for long sequences.
Architectural modification approaches offer fine-grained control and have the potential to generate high-quality outputs aligned with desired attributes. However, they are computationally intensive, require large amounts of attribute-specific training data, and may not be feasible when dealing with multiple levels of text complexity or a wide range of control attributes.

\\\

Another notable approach in the retraining and refactoring category is the Progressive Insertion-based Transformer (POINTER) model introduced by He et al. \cite{he2021parallelrefine}. POINTER aims to enable lexically constrained text generation by modifying the Transformer architecture. Given a set of constrained words, POINTER first generates these words to satisfy the lexical constraints and then iteratively inserts additional words between them until a complete sentence is formed.
The generation process in POINTER is divided into two stages: token-level planning and iterative refinement. In the planning stage, a token-level classifier predicts the positions of the constrained words in the output sequence. This is followed by the iterative refinement stage, where the model inserts words between the constrained tokens in a parallel manner until the sentence is complete. The training objective for POINTER is designed to minimize the cross-entropy loss for both the token-level classifier and the generation model:
\begin{equation}
\label{eq:pointer-objective}
\mathcal{L}{POINTER} = \mathcal{L}{cls} + \mathcal{L}_{gen}
\end{equation}
where $\mathcal{L}{cls}$ is the loss for the token-level classifier, and $\mathcal{L}{gen}$ is the loss for the generation model. The classifier loss is computed using binary cross-entropy, while the generation loss is calculated using standard cross-entropy over the vocabulary.
POINTER achieves high-quality lexically constrained generation by explicitly modeling the planning and refinement process. However, the model assumes that the constrained words are provided as input, which may not always be the case in practical applications. Additionally, the iterative refinement process can be computationally expensive, especially for long sequences.
While retraining and refactoring approaches offer fine-grained control and have the potential to generate high-quality outputs, they are computationally intensive and require large amounts of attribute-specific training data. This can be particularly challenging in specialized domains like medicine, where the availability of suitable training data may be limited. Moreover, training a separate model for each combination of control attributes may not be feasible, especially when dealing with multiple levels of text complexity or a wide range of control attributes.

\\\

Zhang et al. \cite{zhang-etal-2020-pointer} propose POINTER, an insertion-based model for hard-constrained text generation, e.g., including specific words in the output.
POINTER modifies the Transformer architecture to progressively generate constrained words first and then insert finer-grained details iteratively.
While ensuring lexical constraints are met, the approach requires training from scratch on a large corpus and may compromise fluency compared to autoregressive models.
To address factual accuracy in medical text, Contrastive Conditional Text Generation (ConText) \cite{Ye2023} is a relevant approach.

\\\

An alternative is to refactor an existing pre-trained language model to incorporate control mechanisms. 
The POINTER (POINTer-based insertion Transformer) model \cite{zhang2020pointerconstrained} modifies the standard transformer architecture to enable hard-constrained lexical control - i.e. forcing the inclusion or exclusion of specified words and phrases. 
Rather than generating text left-to-right like a typical language model, POINTER generates the constrained tokens first, then iteratively inserts new tokens in the intervals between them. 
The Pointer network is trained to decide where to insert and the Insertion Transformer fills in the missing tokens.
This progressive insertion approach ensures that all lexical constraints are met in the final output. The authors demonstrate POINTER's ability to generate coherent paragraphs from a set of keywords, achieving higher constraint satisfaction than baseline methods like CTRL. 
They also show promising results on dialogue response generation, where POINTER can effectively incorporate dialogue act and style tags.
For our medical QA model, a POINTER-like approach could potentially allow us to constrain generated answers to include certain keywords or phrases while avoiding others, based on the target audience. 
For example, to explain a condition to a patient, we may want to include lay terms and analogies while avoiding medical jargon. When answering a doctor's question, the opposite would be true.
However, the strict constraint satisfaction of POINTER also limits its fluency and diversity compared to standard left-to-right generation. 
The inserter model is less aware of previous context across gaps longer than a certain span. An approach that balances constraint satisfaction with generation fluency would be ideal for medical QA.

\\\

Content control tasks, on the other hand, specifically focus on managing precise text content, such as enforcing the inclusion or exclusion of certain words and phrases. 
POINTER (Progressive Insertion-based Transformer) \cite{zhang_aaai20_pointer} is an early lexical control model using a stepwise, iterative text generation approach. 
While it allows comprehensive control over text, its insertion-based method is inefficient. 
CBART (Constrained BART) \cite{he_emnlp2021_cbart} improves efficiency by dividing the task into two subtasks, where the encoder generates tokens to guide the decoder in parallel prediction.
While retraining methods perform well in tasks requiring strict content control, such as structure control and lexical control, they also have significant drawbacks.
First, they typically require substantial computational resources and time, especially when training large-scale models from scratch. 
Second, to ensure that the model learns the necessary control attributes, a large amount of high-quality, targeted data is needed, further increasing costs. 
These drawbacks make retraining methods less practical when dealing with modern large language models.

\\\

Zhang et al. (2020) introduced POINTER (PrOgressive INsertion-based TransformER) \cite{zhang-etal-2020-pointer}, an insertion-based approach for lexically constrained text generation. 
POINTER modifies the Transformer architecture to generate text progressively by inserting constrained words at a coarse granularity and then refining the generation by inserting finer-grained words between them. 
This iterative process allows the model to generate text that strictly adheres to the given lexical constraints.
While retraining methods can achieve precise control over the generated text, they often require significant computational resources and large-scale attribute-specific datasets. 
This limits their practicality in scenarios where rapid adaptation to new control attributes is necessary or when labeled data is scarce.

\\\

Another example of model refactoring is POINTER (Constrained Progressive Text Generation) \cite{zhang-etal-2020-pointer}, which modifies the transformer architecture to generate text in a progressive manner. 
Given lexical constraints, POINTER first generates the constrained words and then iteratively inserts more detailed words between them until the sentence is complete. 
This approach ensures that the generated text strictly adheres to the provided constraints. 
However, training the model from scratch on a large corpus is computationally expensive, and the fluency of the generated text may not match that of an auto-regressive language model.
While retraining or refactoring methods offer strong controllability, they come with significant computational costs and may require large amounts of attribute-specific training data. 
Moreover, modifying the model architecture can limit its versatility and adaptability to new control attributes.

\\\

Another retraining approach is POINTER \cite{zhang-etal-2020-pointer}, an insertion-based transformer architecture for hard-constrained text generation. 
Unlike traditional left-to-right language models, POINTER generates text by progressively inserting new tokens between existing ones. 
This allows for precise control over the inclusion of specified keywords or phrases in the output. The model is trained from scratch on a large corpus using an insertion-based objective function. 
During inference, POINTER first inserts the target keywords into the input sequence and then fills in the remaining tokens to produce a coherent output.
POINTER offers a promising solution for generating medical text with strict lexical constraints, such as including specific medical terms or phrases. 
However, the insertion-based generation process may struggle to maintain long-range coherence and fluency, especially for longer medical responses. 
Additionally, the model's reliance on keyword insertion limits its ability to control more abstract attributes like text complexity, which often depends on factors beyond vocabulary choice.

\\\

Another example of retraining is the POINTER (Progressive Insertion-based Transformer) model \cite{zhang2020pointerconstrained}. 
POINTER modifies the Transformer architecture to generate text in a progressive manner, first generating the constrained words to satisfy the lexical control conditions, and then iteratively inserting more detailed words between them. 
This approach ensures that the generated sentences meet the specified lexical constraints, albeit at the cost of training the model from scratch on a large-scale corpus.
While retraining methods can produce high-quality controllable text, they often require significant computational resources and large-scale labeled datasets. 
Additionally, modifying the model architecture may limit the model's versatility compared to the original pre-trained model.

# ConText

ConText enhances T5-base with a factual consistency score derived from similarity with relevant evidence snippets.
The model is finetuned using a max-margin loss that rewards factually consistent generations over contrastive negatives.
On medical QA benchmarks like MedQA and PubMedQA, ConText demonstrated improved factual consistency without sacrificing generation quality.
However, ConText's reliance on retrieving relevant evidence snippets could limit its applicability in domains with scarce verified knowledge sources.
While retrain/refactoring methods can yield strong controllability and performance gains, they typically entail substantial computational costs and large-scale training data.
For our medical QA model aiming to control response complexity, retraining an entire model like PubMedBERT \cite{pubmedbert} or BioBERT \cite{biobert} would be prohibitively expensive.

# CoCon

Another notable approach is the Content-Conditioner (CoCon) model introduced by Chan et al. \cite{chan2021coconselfsupecontrolled}.
CoCon achieves fine-grained controllable generation by embedding the control conditions directly into the model's internal states.
The model architecture incorporates a content-conditioning block that learns to fuse the input text with the desired control attributes.
CoCon is trained using self-supervised losses that encourage the model to reconstruct the input text while satisfying the control conditions.
The authors demonstrate CoCon's ability to control sentiment, theme, and other fine-grained attributes in the generated text.
While retraining or refactoring the language model provides a high degree of control over the generated text, it comes with significant computational costs and data requirements.
Training a large-scale language model from scratch necessitates substantial computing resources and a massive amount of attribute-specific training data.
This can limit the practicality of such approaches, especially in domains like healthcare where high-quality labeled data is scarce.

\\\

\citet{chan2021coconselfsupervisedapproach} propose CoCon (Content-Conditioner), which injects a content-conditioning block into a pre-trained GPT model. 
This block takes a content code as additional input and is trained using self-supervised losses to encourage the generation of text that follows the provided content constraints at a fine-grained level. 
By learning to interpolate the content code with the model's hidden states, CoCon can control attributes like sentiment and theme more precisely than CTRL-like approaches, without incurring the cost of a full retraining.
Model refactoring techniques have also been applied successfully to data-to-text generation tasks, such as generating descriptive text from structured tables or knowledge graphs. 
\citet{ribeiro-etal-2021-structural} introduce Structural Adapters, a method to inject graph structure information into pre-trained encoder-decoder models like BART. 
The adapters are trained to encode the graph topology, while the original model parameters are frozen, preserving its pre-trained knowledge. 
This allows the resulting model to produce fluent text that faithfully captures the relationships in the input graph. Such hybrid approaches that combine the power of large pre-trained models with task-specific structural priors are promising for controlling the content and coherence of the generated text.
While model retraining and refactoring offer a great deal of flexibility in defining control attributes and integrating them deeply into the generation process, they can be resource-intensive, especially when applied to large language models. 
Fine-tuning provides a more efficient alternative, as we will see next.

\\\

CoCon (Content-Conditioner) \cite{chan2021coconselfsupervisedapproachcontrolled} addresses the need for finer-grained control by embedding control conditions directly into the internal states of the language model via a CoCon Block. 
This approach reduces training costs by avoiding the need to train models from scratch while providing more precise control compared to CTRL.
In the context of medical text generation, refactoring approaches could be employed to design language models that can inherently adapt to different levels of complexity. 
By incorporating complexity-specific control codes or architectural components, such models could generate outputs tailored to various audiences, from lay readers to medical experts.

\\\

CoCon (Content-Conditioner) \cite{chan2021cocon} presents another refactoring approach that injects a content-conditioning module into a pre-trained language model. 
The CoCon module learns to generate text conditioned on the control attributes through self-supervised losses, enabling word-level and phrase-level content control without the need for large-scale labeled data. 
While this approach is more parameter-efficient compared to full retraining, it still requires architectural modifications and additional training steps.
Adapting CoCon for medical QA would involve designing appropriate self-supervised tasks and control attributes relevant to the medical domain. 
However, the effectiveness of this approach in capturing the nuances of medical language complexity remains to be explored.

\\\

\citeauthor{chan2021coconselfsupervisedapproachcontrolled} (\citeyear{chan2021coconselfsupervisedapproachcontrolled}) introduced the Content-Conditioner (CoCon), a self-supervised approach for controlled text generation that embeds control conditions directly into the language model's internal states.
CoCon employs a novel neural architecture called the CoCon Block, which modulates the model's hidden representations based on the control conditions.
This fine-grained control mechanism enables CoCon to generate text that adheres to specific attributes at the word or phrase level, offering more precise control compared to CTRL.
While retraining methods have demonstrated effectiveness in various controlled text generation tasks, they often require substantial computational resources and large-scale attribute-specific datasets.
This can be a significant limitation in domains like medical text generation, where annotated data is scarce and domain expertise is essential.

\\\

Abstract attribute control tasks aim to guide text generation by steering high-level attributes like sentiment and theme. 
CoCon (Content-Conditioner) \cite{chan2021coconselfsupervisedapproach} addresses the need for fine-grained control by embedding control conditions directly into the internal states of the language model via the CoCon Block. 
This approach not only provides finer control but also reduces training costs by avoiding the need to train models from scratch.

\\\

Refactoring methods, on the other hand, involve modifying the architecture of pre-trained language models to incorporate control mechanisms. 
For example, CoCon \cite{chan2021coconsupervisedapproach} introduces a content-conditioner module that is trained to generate content-relevant representations based on the input context and control attributes. 
These representations are then integrated into the language model through a modified attention mechanism, allowing for fine-grained control over the generated text. 
However, the effectiveness of refactoring methods in the medical domain remains largely unexplored, and their ability to handle the complexities of medical language and maintain factual accuracy requires further investigation.
In the context of developing a controllable medical language model for QA, retraining or refactoring methods may be considered when fine-tuning pre-trained models proves insufficient for achieving the desired level of control over the complexity of medical language. 
However, the scarcity of large-scale medical corpora with explicit complexity annotations and the potential challenges in maintaining factual accuracy while manipulating the model architecture present significant obstacles. 
Therefore, careful evaluation and adaptation of these methods would be necessary to ensure their suitability for the specific requirements of medical QA.

\\\

CoCon (Content-Conditioner) \cite{chan2021coconselfsupervisedapproach} addresses this issue by introducing a content-conditioning block into a pre-trained GPT model. 
The block is trained using self-supervised losses to incorporate content-level control without modifying the base model. 
This approach achieves fine-grained control over entities and phrases while being more computationally efficient than training from scratch.
In the context of a controllable medical language model, retraining or refactoring approaches could be used to incorporate medical domain knowledge and control codes for different levels of text complexity. However, the lack of large-scale medical corpora with appropriate control annotations may limit the applicability of these methods. Additionally, the computational cost of retraining large language models on medical data could be prohibitive.

\\\

Building upon the idea of CTRL, \citet{chan2021cocontentcontrollertextgeneration} introduced the Content-Conditioner (CoCon) model, which allows for fine-grained content control at the word and phrase level.
CoCon injects a content conditioning block into the transformer architecture and uses self-supervised losses to train the model to generate text that incorporates the provided content selectively.
This approach enables more precise control over the generated text compared to CTRL, which operates at a coarser level of control codes.
In the context of medical text generation, \citet{shen2022mredmetareviewdataset} proposed the Medical Review Dataset (MReD), a large-scale dataset of medical reviews with structured annotations of key information such as the patient's medical history, symptoms, and treatments.
They trained a transformer-based model called MReD-Gen on this dataset, which can generate medically coherent and informative review summaries conditioned on the structured annotations.
While MReD-Gen is not explicitly designed for controllable generation, the structured annotations provide a form of content control that could be adapted for generating medical text with varying levels of complexity.
However, retraining or refactoring methods have some limitations.
Training a new model from scratch or significantly modifying an existing model can be computationally expensive and time-consuming, especially for large-scale LLMs.
Moreover, these methods often require a large amount of attribute-specific training data, which may not always be available, particularly for specialized domains like medicine.
Nevertheless, retraining or refactoring can be a powerful approach when fine-grained control over the model architecture and training process is desired, and sufficient computational resources and training data are available.

\\\

CoCon (Content-Conditioner) \cite{chan2021coconselfsupervisedapproachcontrolled} addresses more fine-grained controllability by introducing a content-conditioning block within a pre-trained GPT model. 
The CoCon block takes the control code as a separate input and is trained using self-supervised losses to encourage the model to generate text that incorporates the conditioned content. 
This approach allows for more precise control at the word and phrase level, which could be beneficial for generating medical text with specific terminology or factual requirements. 
However, CoCon still requires fine-tuning the entire GPT model, which can be computationally expensive, especially with larger models.

\\\

Another example of refactoring is the CoCon (Content-Conditioner) model proposed by Chan et al. \cite{chan2021coconselfsupervisedapproachcontrollable}.
CoCon introduces a content-conditioning block into the GPT architecture, which takes the control attributes as a separate input and learns to guide the model's output towards the desired properties.
The model is trained using self-supervised objectives, such as reconstructing the input text and enforcing attribute consistency, without the need for labeled data.
CoCon demonstrates the potential of refactoring existing architectures to incorporate control mechanisms while leveraging unsupervised learning techniques.
While retraining or refactoring approaches offer a high degree of control over the generated text, they come with significant computational costs and may require substantial modifications to the model architecture.
Furthermore, these methods may not always be feasible or practical, especially when working with large-scale pre-trained models or in resource-constrained environments.

\\\

CoCon (Content-Conditioner) \cite{chan2021coconselfsuper} injects a content conditioning block into a pre-trained GPT model, which learns to embed the control conditions directly into the model's hidden states. 
This allows more precise control at the word and phrase level without the need for full retraining. 
Similarly, GeDi (Generative Discriminator) \cite{krause2021gedigenerativediscr} trains a class-conditional language model (CC-LM) to guide generation from a base model like GPT-2 or GPT-3 toward desired attributes. 
The CC-LM is fine-tuned to maximize the likelihood of the desired attribute during generation, acting as a "discriminator" to steer the base model.
While model retraining and refactoring approaches have shown promise for controlled text generation, their applicability to the medical domain may be limited by the lack of large-scale, attribute-labeled medical text corpora. 
Collecting such datasets, especially with fine-grained attributes like complexity levels suitable for different audiences, can be challenging and resource-intensive. 
Moreover, retraining large language models from scratch, as in CTRL, may be computationally prohibitive for many researchers and practitioners.

\\\

CoCon (Content Conditioner) \cite{chan2021coconselfsupevisedapproach} is another notable example that refactors the GPT architecture by introducing a conditional content modeling block.
This block consists of a cross-attention mechanism that attends to the concatenated content representation and control code, enabling more precise control at the word- and phrase-level.
CoCon is trained using self-supervised losses to minimize the need for labeled data.
Experiments show that CoCon can effectively incorporate target content while maintaining output fluency and diversity.
Retraining or refactoring approaches offer fine-grained controllability but often require significant computational resources and large-scale attribute-specific data.
For biomedical applications, curating such datasets with desired complexity levels and medical fidelity can be challenging and time-consuming.
Moreover, retraining from scratch forgoes the benefits of large-scale pre-training which has proven crucial for language understanding and generation in technical domains like medicine \cite{gu2022domainspecificlanguagemodel, beltagy2019scibert}.

\\\

A more efficient alternative to retraining is refactoring, which involves modifying specific components of a pre-trained language model while preserving its core architecture. 
For instance, the CoCon (Content-Conditioner) framework \cite{chan2021coconselfsupervisedapproach} introduces a content-conditioning block into the GPT-2 architecture to enable fine-grained control over the generated text. 
The content-conditioning block learns to inject attribute-specific information into the model's hidden representations using a self-supervised learning objective. 
At inference time, the block can be used to guide the generation towards the desired attributes without the need for retraining the entire model.
CoCon's modular design makes it a promising candidate for controlled medical text generation. 
By training the content-conditioning block on a medical corpus with complexity-related attributes, the model could learn to generate text at different levels of technicality. 
However, the effectiveness of this approach depends on the quality and diversity of the attribute-specific medical data used for training. 
Furthermore, integrating the content-conditioning block into existing medical language models may require non-trivial modifications to their architectures.
In summary, while retraining and refactoring methods have shown success in controlled text generation, their application to medical language models faces challenges related to data availability, computational resources, and the specialized nature of medical text. 
Nonetheless, these approaches offer valuable insights into designing architectures that can effectively incorporate control attributes into the generation process.

\\\

CoCon \cite{chan2021coconsupervisedapproachcontrollable} takes a different approach by introducing a conditional control module into an existing pre-trained model (GPT). 
The control block is trained using self-supervised losses to learn fine-grained content control at word and phrase levels. 
This allows for more precise control over generated text while leveraging the pre-trained model's knowledge. 
However, CoCon still requires modifying the model architecture and training the control block, which can be resource-intensive.
These retraining/refactoring approaches offer strong controllability but may be less practical for large-scale pre-trained models due to computational costs and the need for extensive labeled data.
They may also be less flexible in adapting to new control attributes, as architectural changes are often task-specific.

\\\

To address this limitation, Chan et al. (2021) proposed the Content-Conditioner (CoCon) model, which embeds control conditions directly into the language model's internal states via a dedicated conditioning block \cite{chan2021coconselfsupervisedapproach}. 
By conditioning on both input context and attribute representations, CoCon achieves finer-grained control without the need to train from scratch, reducing computational costs. 
This approach could be particularly relevant for tailoring medical text complexity, as it allows for more precise adjustments based on the target audience's expertise level.
However, retraining methods often require substantial computational resources and large-scale attribute-specific datasets, making them less practical for rapid deployment or resource-constrained environments. 
Fine-tuning pre-trained models, discussed in the next section, offers a more efficient alternative.

# Dialogue-CRF

Zhang et al. \cite{zhang2022dialoguecrf} proposed Dialogue Conditional Random Field (Dialogue-CRF), a conditional language model that explicitly models the dependencies between dialogue attributes and generated responses. The Dialogue-CRF model is trained on a large-scale dialogue dataset with annotated attributes such as dialogue acts, emotions, and topics. By learning the attribute-response relationships through a CRF-based objective, the model can generate responses conditioned on the specified attributes, enabling controlled dialogue generation.
While retraining or refactoring approaches offer a high degree of control over the generated text, they often require significant computational resources and large-scale attribute-specific datasets. This can limit their practicality in scenarios where labeled data is scarce or when adapting to new control attributes is necessary.



