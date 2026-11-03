# User and Academic Demonstration Guide

**Project Title:** AI-Powered Semantic Search Engine Using LLMs and Attention Mechanisms  
**Audience:** AI Engineers, Data Analysts, Evaluators, and Machine Learning Practitioners

---

## 1. Quick Start Walkthrough

### 1.1 Launching the Application
Execute the following commands in Windows PowerShell:
```powershell
# Navigate to project directory
cd c:\Users\balut\OneDrive\Desktop\NLP

# Activate the virtual environment
.\.venv\Scripts\activate

# Run the Flask web application
python app.py
```
Open your web browser and navigate to: **`http://127.0.0.1:5000`**

---

## 2. Page-by-Page Feature Exploration

### 2.1 Search Dashboard (`/`)
1. **Search Bar**: Enter natural-language queries (e.g., *"How can AI detect diseases early in healthcare?"*).
2. **Search Mode Toggle**:
   - **Hybrid**: Balances keyword precision with semantic generalization.
   - **Semantic**: Uses Sentence Transformers to find conceptually similar passages even if verbatim words differ.
   - **TF-IDF**: Scikit-Learn sparse vector space retrieval.
   - **Inverted Index**: Classic positional postings list retrieval.
3. **Sliders**:
   - **Top-K Slider**: Choose between 1 and 15 results.
   - **Alpha Slider**: Adjust $\alpha$ from 0.0 (100% semantic) to 1.0 (100% keyword).
4. **Results Cards**:
   - Note the score breakdown (Keyword vs. Semantic vs. Final Score).
   - Click *"Ask AI about this"* to carry the query directly into the RAG engine.

### 2.2 AI Question Answering (RAG) (`/qa`)
1. Enter an inquisitive question (e.g., *"Explain the mathematical formulation of Scaled Dot-Product Attention."*).
2. Click **Generate Answer**.
3. Observe:
   - **Grounded Answer**: Generated strictly from retrieved document chunks.
   - **Verified Source Citations**: Direct references to document titles and page numbers.
   - **Insufficient Evidence Detector**: Try asking an out-of-domain question (e.g., *"What is quantum gravity?"*); the system alerts that no evidence exists in the corpus rather than hallucinating!

### 2.3 Document Upload (`/upload`)
1. Drag and drop any `.pdf` or `.txt` file onto the dropzone.
2. The system executes:
   - File size validation (up to 16 MB).
   - SHA-256 deduplication (try re-uploading the same file to observe the duplicate alert).
   - PyMuPDF text extraction.
   - Sliding-window passage chunking (200 words with 40-word overlap).
   - Instant dual-indexing into both TF-IDF and dense embeddings.

### 2.4 Document Management (`/documents`)
1. View the complete table of indexed documents, word counts, and passage chunk counts.
2. Click **View Details** (`/documents/<id>`) to inspect individual passage chunks.
3. Click **Summarize** to generate an on-the-fly executive summary of any document.
4. Click **Delete** to test cascading deletion across the SQLite database and search indexes.

### 2.5 Transformer Attention Heatmap (`/attention`)
1. Enter any natural language sentence (e.g., *"Artificial intelligence detects disease early in healthcare analytics."*).
2. Select **Transformer Layer** (Layer 1 to 6) and **Attention Head** (Head 1 to 12, or Average).
3. Click **Extract Attention**.
4. Observe the real attention heatmap generated via Matplotlib and inspect the token-to-token compatibility matrix.

### 2.6 Word2Vec Laboratory (`/word2vec`)
1. Select between **Continuous Bag-of-Words (CBOW)** and **Continuous Skip-gram**.
2. Calculate pairwise word cosine similarity (e.g., *"disease"* vs. *"healthcare"*).
3. Query nearest neighbor words in vector space.
4. Review the educational comparison matrix comparing One-Hot vs. TF-IDF vs. Word2Vec vs. Transformers.

### 2.7 Document Clustering & PCA (`/clustering`)
1. Select cluster count $K$ (from 2 to 8).
2. Switch feature representation between **TF-IDF** and **Dense Embeddings**.
3. View the 2D PCA scatter plot on the interactive HTML5 canvas.
4. Inspect the **Silhouette Score** and top extracted topic keywords per cluster.

### 2.8 Text Visualizations & Linguistic Graphs (`/visualizations`)
1. Click **Visualizations** in the top navigation bar or from any document details page.
2. **Document Scope Selector**:
   - Choose **Entire Ingested Corpus** to analyze all documents combined, or select any individual document (e.g. *Early Detection of Chronic Kidney Disease*).
3. **Word Cloud**:
   - High-resolution visual representation where word size maps directly to term frequency after stopword removal.
   - Inspect the companion table listing the top 20 vocabulary items and their exact counts.
4. **Bigram Co-occurrence Network Graph**:
   - Built with NetworkX and Matplotlib using spring layout.
   - Nodes represent vocabulary words; edges represent consecutive transitions ($w_1 \to w_2$). Edge thickness and labels highlight frequent collocations like *machine → learning*, *deep → learning*, and *kidney → disease*.
5. **Frequency Bar Chart**:
   - Switch between **Unigrams (single words)** and **Bigrams (word pairs)**.
   - Horizontal bars rank top items with exact numerical frequency labels.

### 2.9 Analytics & Empirical Evaluation (`/analytics`)
1. View real search metrics, document distribution, and query latency history.
2. Click **Re-Run Benchmark Evaluation** to execute the 8 ground-truth benchmark queries.
3. View the comparative evaluation table contrasting Precision@5, Recall@5, MRR, and Latency.

### 2.10 Settings & System Configuration (`/settings`)
1. Change default retrieval modes and alpha weights.
2. Configure LLM providers: Local Ollama, OpenAI-compatible API, or Extractive Fallback.
3. Rebuild search indexes or reset sample corpus with one click.

---

## 3. Academic Presentation Tips for B.Tech Evaluators

1. **Highlight the Vocabulary Mismatch Problem**:
   - Query: *"diagnosing patient sickness"*
   - In **TF-IDF mode**, it returns low or zero hits because the documents use *"disease detection"* and *"clinical analytics"*.
   - In **Semantic mode**, it immediately retrieves the chronic kidney and cardiovascular documents at Rank #1.
2. **Demonstrate Hybrid Superiority**:
   - Show how Hybrid search captures both exact keywords (e.g., *"eGFR"*, *"YOLOv8"*, *"MITRE"*) and broader conceptual queries.
3. **Showcase Real Attention Matrices**:
   - Point out that the attention weights are mathematical tensors extracted directly from transformer encoders, not mock data.
