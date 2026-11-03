# Algorithmic & Mathematical Formulations

This document provides rigorous mathematical definitions for all information retrieval, embedding, attention, clustering, and evaluation algorithms implemented in this project.

---

## 1. Traditional Retrieval: TF-IDF & Cosine Similarity

### 1.1 Term Frequency - Inverse Document Frequency
For a term $t$ in document or passage $d$ within collection $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- **Sublinear Term Frequency**:
  $$\text{TF}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{for } \text{count}(t, d) > 0$$
- **Smoothed Inverse Document Frequency**:
  $$\text{IDF}(t, D) = \ln\left( \frac{1 + |D|}{1 + \text{DF}(t)} \right) + 1$$
  where $\text{DF}(t)$ is the number of documents containing term $t$.

### 1.2 Cosine Similarity
Given sparse query vector $\mathbf{q}$ and document vector $\mathbf{d}$ in $\mathbb{R}^{|V|}$:

$$\text{CosineSimilarity}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2} = \frac{\sum_{i=1}^{|V|} q_i d_i}{\sqrt{\sum_{i=1}^{|V|} q_i^2} \sqrt{\sum_{i=1}^{|V|} d_i^2}}$$

---

## 2. Distributed Word Embeddings: Word2Vec

### 2.1 Continuous Bag-of-Words (CBOW)
CBOW maximizes the log probability of predicting a center word $w_t$ given context words $w_{t-c}, \dots, w_{t+c}$:

$$\mathcal{L}_{\text{CBOW}} = \sum_{t=1}^{T} \log P(w_t \mid w_{t-c}, \dots, w_{t+c})$$

Where the average context vector $\bar{\mathbf{v}}_c = \frac{1}{2c} \sum_{-c \le j \le c, j \ne 0} \mathbf{v}_{w_{t+j}}$ and:

$$P(w_t \mid \bar{\mathbf{v}}_c) = \frac{\exp(\mathbf{v}'_{w_t}{}^\top \bar{\mathbf{v}}_c)}{\sum_{w \in V} \exp(\mathbf{v}'_w{}^\top \bar{\mathbf{v}}_c)}$$

### 2.2 Continuous Skip-Gram
Skip-gram maximizes the probability of predicting context words given the center word $w_t$:

$$\mathcal{L}_{\text{Skip-gram}} = \sum_{t=1}^{T} \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)$$

---

## 3. Dense Semantic Search: Sentence Transformers

A bi-encoder maps an input passage $d$ into a fixed-dimensional dense vector $\mathbf{e}_d \in \mathbb{R}^{384}$:

$$\mathbf{e}_d = \text{MeanPooling}(\text{TransformerEncoder}(d))$$

Normalized semantic similarity between query embedding $\mathbf{e}_q$ and passage embedding $\mathbf{e}_d$:

$$\text{Sim}_{\text{semantic}}(q, d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\|_2 \|\mathbf{e}_d\|_2}$$

---

## 4. Hybrid Search Fusion Formulations

### 4.1 Linear Combination with Min-Max Calibration
Given raw scores $S_{\text{kw}}(d)$ and $S_{\text{sem}}(d)$:

$$S'_{\text{kw}}(d) = \frac{S_{\text{kw}}(d) - \min(S_{\text{kw}})}{\max(S_{\text{kw}}) - \min(S_{\text{kw}}) + \epsilon}$$

$$\text{FinalScore}(d) = \alpha \cdot S'_{\text{kw}}(d) + (1 - \alpha) \cdot S'_{\text{sem}}(d)$$

Where $\alpha \in [0.0, 1.0]$ is the configurable keyword-semantic trade-off coefficient.

### 4.2 Reciprocal Rank Fusion (RRF)
Combines distinct rank orders without requiring score calibration:

$$\text{RRF\_Score}(d) = \sum_{m \in M} \frac{1}{k_{\text{rrf}} + \text{rank}_m(d)}$$

Where $k_{\text{rrf}} = 60$ acts as a ranking smoothing constant.

---

## 5. Transformer Attention Mechanisms

### 5.1 Scaled Dot-Product Attention
Given Query matrix $Q \in \mathbb{R}^{n \times d_k}$, Key matrix $K \in \mathbb{R}^{m \times d_k}$, and Value matrix $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V$$

Where:
- $\frac{1}{\sqrt{d_k}}$ is the temperature scaling factor preventing vanishing gradients in high dimensions.
- The attention weight matrix $A_{i,j}$ represents the fraction of context routing from token $i$ to token $j$:

$$A_{i,j} = \frac{\exp\left( \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} \right)}{\sum_{l=1}^{m} \exp\left( \frac{\mathbf{q}_i \cdot \mathbf{k}_l}{\sqrt{d_k}} \right)}$$

### 5.2 Multi-Head Attention
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

---

## 6. Document Clustering & Dimensionality Reduction

### 6.1 K-Means Objective Function
Partitions $N$ document vectors into $K$ disjoint clusters $C = \{C_1, \dots, C_K\}$ minimizing within-cluster sum of squares:

$$J = \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|_2^2$$

Where $\boldsymbol{\mu}_k$ is the cluster centroid: $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}$.

### 6.2 Silhouette Coefficient
For document $i \in C_A$:
- $a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \ne i} d(i, j)$ (mean intra-cluster distance)
- $b(i) = \min_{B \ne A} \frac{1}{|C_B|} \sum_{j \in C_B} d(i, j)$ (mean nearest-cluster distance)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$

---

## 7. Information Retrieval Evaluation Metrics

### 7.1 Precision@K
$$\text{Precision@K} = \frac{|\text{Relevant Documents Retrieved in Top } K|}{K}$$

### 7.2 Recall@K
$$\text{Recall@K} = \frac{|\text{Relevant Documents Retrieved in Top } K|}{|\text{Total Relevant Documents in Corpus}|}$$

### 7.3 Mean Reciprocal Rank (MRR)
For a query set $Q$:

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Where $\text{rank}_i$ is the 1-based rank position of the first relevant document retrieved for query $i$.

---

## 8. Linguistic Graphs & Frequency Visualizations

### 8.1 Word Cloud Frequency Scaling
Term sizes are proportionally scaled to empirical term frequencies $f(w)$:

$$\text{FontSize}(w) = S_{\min} + (S_{\max} - S_{\min}) \cdot \left(\frac{f(w) - \min_v f(v)}{\max_v f(v) - \min_v f(v)}\right)^\gamma$$

where $\gamma \approx 1$ provides linear scaling across non-stopword tokens.

### 8.2 Bigram Co-occurrence Network Graph
Given adjacent token sequences $(w_i, w_{i+1})$, a weighted undirected graph $G = (V, E, W)$ is constructed:
- $V = \{w \mid w \text{ is a frequent non-stopword token}\}$
- $E = \{(u, v) \mid u \text{ and } v \text{ occur consecutively in text}\}$
- Edge weight $W(u, v) = \text{count}(u, v) + \text{count}(v, u)$

Node positions are calculated using the Fruchterman-Reingold spring force algorithm:

$$F_{\text{repulsive}}(d) = \frac{k^2}{d}, \quad F_{\text{attractive}}(d) = \frac{d^2}{k}$$

where $k = C \sqrt{\frac{\text{Area}}{|V|}}$ determines optimal node spacing. Higher edge weights pull co-occurring concepts closer together, revealing domain clusters.
