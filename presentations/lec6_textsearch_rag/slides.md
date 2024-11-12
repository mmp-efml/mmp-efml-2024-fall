---
marp: true
paginate: true
math: true
---

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
section::after {
  content: attr(data-marpit-pagination) '/' attr(data-marpit-pagination-total);
}
</style>

# Text Embedding. Text Retrieval & Ranking. Retrieval-Augmented Generation.

###### Alekseev Ilya, EFML, Fall 2024.

---

# Text Embedding

## Outline

- Text Embedding & Sequence-level Tasks
- BERT Embedding
- SBERT
- Contrastive Learning: Loss, Positives, Negatives

---

## Text Embedding

*Embedding must be useful as **feature representation** and for **vector search***.

![center width:900](figures/vectorization.jpg)

---

## Sequence-level Tasks

- **Natural Language Inference (NLI)**: contradiction, eintailment, and neutral (pair classification)
- **Bitext Mining**: mine closest translation pairs from parallel corpus (knn)
- **Semantic Textual Similarity (STS)**: estimate the similarity of two texts (pair regression)
- **Retrieval**: find relevant documents for query text (knn)
- **Parahpase detection** (pair classification)
- Classification, clustering, reranking, summarization

---

## BERT Embedding

Feed to BERT and pool last hidden states:
- CLS
- Average
- Attention

![bg right:60% width:550](figures/cls-embedding.drawio.svg)

---

# BERT is not trained to produce good embeddings!

![center width:900](figures/encoder-bert.drawio.svg)

---

# We need **sentence-level** task to encourage model to aggregate info effectively

Reimers & Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks", EMNLP 2019 (citations: 12866)

Sequence-level task: *train BERT on NLI data*.

---

## SBERT

![center width:900](figures/sbert.drawio.svg)

---

## SBERT: pros and cons

➕ sequence-level task
➕ bottleneck trick
➖ supervised data
➖ only features but not a vector search 

---

## Contrastive Learning

$$
\mathcal{L} = -\log\frac{\exp({\cos (x, y)})} {\sum_{z\in Z} \exp{(\cos(x, z))}}
$$

![bg right height:500](figures/project-visualization.drawio.svg)


---

## How to Mine Positives

- supervised datasets (NLI, STS, summarization, retrieval)
- scrapped data (QA forums, Reddit threads, web articles, news)
- augmentations (synonyms, paraphasing, dropout, token shuffling)

---

## How to Mine Negatives

- in-batch negative sampling
- queue
- memory bank
- momentum contrast (MoCo)

---

## In-batch Negative Sampling

```python
# joint embedding
x_emb = encoder(x_txt)  # [B, d]
y_emb = encoder(y_txt)  # [B, d]

# pairwise cosine similarities
x_emb = F.normalize(x_emb, dim=1)
y_emb = F.normalize(y_emb, dim=1)
similarities = x_emb @ y_emb.T  # [B, B]

# symmetric loss
labels = torch.arange(len(x_emb))
loss_r = F.cross_entropy(similarities, labels, reduction='mean')
loss_c = F.cross_entropy(similarities.T, labels, reduction='mean')
loss = (loss_c + loss_r) / 2
```

---

![bg width:350](figures/in-batch.drawio.svg)

![bg width:350](figures/large-batch.drawio.svg)

![bg width:350](figures/hard-negatives.drawio.svg)

---

## SOTA Embedding Models

https://huggingface.co/spaces/mteb/leaderboard

---

## Text Embedding: Summary

- Text Embedding & Sequence-level Tasks
- BERT Embedding
- SBERT
- Contrastive Learning: Loss, Positives, Negatives

---

# Text Retrieval & Ranking

## Outline

- Symmetric vs Asymmetric Search
- Bi-encoder vs Cross-encoder
- Sparse Text Embedding: BM25

---
## Retrieval Types

- **symmetric** search (clustering, knn, bitext mining)
    - query $\sim$ document
    - `q="The last time the survey was conducted, in 1995, those numbers matched."`
    - `d="In 1978, the paper's numbers weren't believed to be true."`
- **asymmetric** search (web search, QA)
    - query $\not\sim$ document
    
    - `q="What is Python"`
    - `d="Python is an interpreted, high-level and general-purpose programming language."`

---

## Search Engine


![center](figures/retrieve-rerank.png)

---

## Bi-encoder

![center width:900](figures/encoder-bi-encoder.drawio.svg)

---

## Cross-encoder

![center width:900](figures/encoder-cross-encoder.drawio.svg)

---

## Sparse Text Embedding

Вектор $e(d)$ размера $|V|$:

- BoW:
$$
[e(d)]_i=\text{tf}(w_i,d)
$$
- TF-IDF
$$
[e(d)]_i=\text{tf}(w_i,d)\cdot\text{idf}(w_i)
$$

- BM25
$$
[e(d)]_i=\widetilde{\text{tf}}(w_i,d)\cdot\widetilde{\text{idf}}(w_i)
$$

---

## TF-IDF

- term frequency $\text{tf}(w,d)$ есть число вхождений токена $w_i$ в документ $d$
- document frequency $\text{df}(w)$ есть число документов, в которых встречается $w$
- inverse document frequency есть мера редкости токена:
$$
\text{idf}(w)=1+\log{1+|D|\over1+\text{df}(w)}
$$
- вместе дает число токенов с учётом редкости каждого токена:
$$
[e(d)]_i=\text{tf}(w_i,d)\cdot\text{idf}(w_i)
$$

---

## BM25

- пусть $\ell(d)$ это отношение длины $d$ к средней длине документов в датасете
- term frequency с поправкой на длину документа:
$$
\widetilde{\text{tf}}(w,d)={3\cdot\text{tf}(w)\over 3(0.25+0.75\cdot\ell(d))+\text{tf}(w)}
$$
- inverse document frequency
$$
\widetilde{\text{idf}}(w_i)=\log{|D|-\text{df}(w)+0.5\over\text{df}(w)+0.5}
$$
- вместе это дает число токенов с учетом редкости, длины текста, числа повторений этого токена
$$
[e(d)]_i=\widetilde{\text{tf}}(w_i,d)\cdot\widetilde{\text{idf}}(w_i)
$$

---

## Text Retrieval & Ranking: Summary

- Symmetric vs Asymmetric Search
- Search Engine Pipeline
- Bi-encoder vs Cross-encoder
- Sparse Text Embedding: BM25

---

# Retrieval Augmented Generation

## Outline

- Introduction. Naive RAG
- Evaluation
- Improve RAG. Prompting Techniques
- Improve RAG. Retriever and LLM Joint Training

---

# Introduction. Naive RAG

---

## Language Models are Few-Shot Learners 

GPT-3 [Brown et al., 2020]

> ...tasks which require using the information stored in the model’s parameters to answer general knowledge questions.

![bg right:40% width:700](figures/next-token-prediction.png)

---


## QA via Prompt Completion

Problems of simple generation:
- hallucinations, missing references
- hard to update

<!-- ![bg right:50% width:600](figures/completion.jpeg) -->

![bg right width:600](figures/my-completion.png)


Solution: **retrieval-augmented generation (RAG)**

---

## Naive RAG

- retrieve documents relevant to query (user's input)
- insert top-k documents into prompt
- feed as prompt

![bg right width:600](figures/rag-completion.png)

---

## Knowledge Stores

- web pages
- PDF, word, markdown (closed-book)
- wikipedia dump
- search engines

---

## Implementation: LangChain

![center width:1000](figures/vector-database.png)

---

## Implementation: LangChain

- choose SOTA LLM (Chatbot Arena)
- choose SOTA embedder (MTEB)

See also: [LlamaIndex](https://docs.llamaindex.ai/en/stable/index.html).

![bg right:40% width:400](figures/langchain-qa.png)

---

# Evaluation

- Knowledge-intensive tasks
- RAG-specific benchmarks

---

## RAGAs [Es et al., 2023]

Automated evaluation with LLM as assesor.

![center width:900](figures/ragas.drawio.svg)

---

## RAGAs [Es et al., 2023]

![center ](figures/ragas.png)

---

## Improve Your RAG

- choose SOTA embedder and LLM according to public leaderboards
- prompting techniques
- train retriever and generator jointly

---

## Prompting Techniques

- query summarization / paraphrasing 
- decompose into multiple questions (Chain-of-Thought)
- reranking chunks

---

## Prompting Techniques: Chain-of-Thought

![center](figures/chain-of-thought.png)

---

## Prompting Techniques: Rerank Chunks

Lost in the middle [Liu et al., 2023]

![bg right width:500](figures/lost-in-the-middle.png)

---

## Train Retriever and Generator Jointly

- RAG [Piktus et al., 2020], BART + DPR
- Hindsight [Paranjape et al., 2022], BART + Colbert
- Atlas [Izacard et al., 2022], T5 + Contriever
- Replug [Shi et al., 2023], GPT-3 + Contriever
- RA-DIT [Lin et al, 2023], Llama 2 + DRAGON

Meta AI almost everywhere!

---

## RAG: Summary

- Introduction. Naive RAG
- Evaluation
- Improve RAG. Prompting Techniques
- Improve RAG. Retriever and LLM Joint Training
