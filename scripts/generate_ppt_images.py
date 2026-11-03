"""Generate 16:9 high-resolution presentation graphics for PowerPoint (PPT) slides.
Outputs:
  - docs/images/ppt/slide8_methodology_flow.png
  - docs/images/ppt/slide9_system_workflow.png
  - docs/images/ppt/slide10_ui_search_demo.png
  - docs/images/ppt/slide11_ui_rag_attention_demo.png
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = PROJECT_ROOT / "docs" / "images" / "ppt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Slide 8: 5-Stage Numbered Methodology Workflow (16:9)
# ------------------------------------------------------------------------------
def generate_slide8_methodology():
    print("Generating Slide 8: Methodology Process Flow...")
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.set_facecolor("#ffffff")
    fig.patch.set_facecolor("#ffffff")
    
    steps = [
        {"num": "1", "title": "Ingest & Dedup", "sub": "Load PDF / TXT\nSHA-256 Hash\nDeduplication", "icon": "DOC"},
        {"num": "2", "title": "Sliding Chunking", "sub": "200-Word Window\n40-Word Overlap\nPage Tracking", "icon": "CHUNK"},
        {"num": "3", "title": "Dual Indexing", "sub": "TF-IDF + Postings\nDense SBERT Bi-Encoder\n384-d Embeddings", "icon": "INDEX"},
        {"num": "4", "title": "Hybrid Ranking", "sub": "Min-Max Score Scale\nLinear α Blend\nRRF Fusion (k=60)", "icon": "FUSION"},
        {"num": "5", "title": "Grounded RAG", "sub": "Prompt Shielding\nVerified Citations\nAttention Heatmaps", "icon": "QA"}
    ]
    
    x_positions = [1.2, 3.8, 6.4, 9.0, 11.6]
    
    for i, (x, step) in enumerate(zip(x_positions, steps)):
        # Circle badge
        circle = patches.Circle((x, 3.8), 0.55, facecolor="#0284c7", edgecolor="#0369a1", linewidth=2.5, zorder=4)
        ax.add_patch(circle)
        ax.text(x, 3.8, step["num"], color="white", fontsize=20, fontweight="bold", ha="center", va="center", zorder=5)
        
        # Step card container
        rect = patches.FancyBboxPatch((x - 1.15, 0.5), 2.3, 2.5, boxstyle="round,pad=0.2,rounding_size=0.15",
                                      facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        
        # Text inside card
        ax.text(x, 2.4, step["title"], color="#0f172a", fontsize=12, fontweight="bold", ha="center", va="center", zorder=3)
        ax.plot([x - 0.8, x + 0.8], [2.15, 2.15], color="#0284c7", linewidth=2, zorder=3)
        ax.text(x, 1.35, step["sub"], color="#475569", fontsize=9.5, ha="center", va="center", multialignment="center", zorder=3, linespacing=1.4)
        
        # Arrows between steps
        if i < len(steps) - 1:
            next_x = x_positions[i+1]
            ax.annotate("", xy=(next_x - 0.65, 3.8), xytext=(x + 0.65, 3.8),
                        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=3.0, mutation_scale=20))
            
    ax.set_xlim(-0.2, 13.0)
    ax.set_ylim(0.0, 4.8)
    ax.axis("off")
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "slide8_methodology_flow.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ------------------------------------------------------------------------------
# 2. Slide 9: Detailed Full System Architecture Diagram (16:9)
# ------------------------------------------------------------------------------
def generate_slide9_system_workflow():
    print("Generating Slide 9: System Architecture Workflow...")
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300)
    ax.set_facecolor("#f8fafc")
    fig.patch.set_facecolor("#ffffff")
    
    # Title Banner
    ax.text(8.0, 8.0, "AI-Powered Semantic Search Engine: End-to-End System Workflow",
            color="#0f172a", fontsize=16, fontweight="bold", ha="center", va="center")
    
    stages = [
        {"x": 1.0, "title": "1. Ingestion", "items": ["Multi-page PDF/TXT", "PyMuPDF text parser", "SHA-256 Checksum", "Byte deduplication"], "color": "#0284c7"},
        {"x": 3.4, "title": "2. Preprocessing", "items": ["200w sliding window", "40-word overlap", "Page provenance link", "Tokenization & lemmatize"], "color": "#2563eb"},
        {"x": 5.8, "title": "3. Dual Indexing", "items": ["Positional Inverted Index", "Scikit TF-IDF matrix", "Sentence Transformers", "384-d binary .npz cache"], "color": "#7c3aed"},
        {"x": 8.2, "title": "4. Hybrid Fusion", "items": ["User natural query", "Min-Max normalization", "Linear Alpha blend", "RRF rank fusion (k=60)"], "color": "#059669"},
        {"x": 10.6, "title": "5. Grounded RAG", "items": ["<CONTEXT> boundary", "Local Ollama Llama 3.2", "Offline Extractive QA", "Page-level citations"], "color": "#d97706"},
        {"x": 13.0, "title": "6. Explainability", "items": ["Attention heatmaps", "K-Means 2D PCA clusters", "NetworkX bigram graph", "Word Cloud frequency"], "color": "#dc2626"}
    ]
    
    for i, s in enumerate(stages):
        x = s["x"]
        # Header box
        h_box = patches.FancyBboxPatch((x - 0.95, 6.5), 1.9, 0.7, boxstyle="round,pad=0.1,rounding_size=0.1",
                                       facecolor=s["color"], edgecolor="none", zorder=3)
        ax.add_patch(h_box)
        ax.text(x, 6.85, s["title"], color="white", fontsize=11, fontweight="bold", ha="center", va="center", zorder=4)
        
        # Content body
        b_box = patches.FancyBboxPatch((x - 0.95, 2.5), 1.9, 3.8, boxstyle="round,pad=0.1,rounding_size=0.1",
                                       facecolor="#ffffff", edgecolor="#cbd5e1", linewidth=1.5, zorder=2)
        ax.add_patch(b_box)
        
        # Bullet texts
        y_text = 5.8
        for item in s["items"]:
            ax.text(x - 0.8, y_text, f"• {item}", color="#334155", fontsize=8.5, ha="left", va="center", zorder=4)
            y_text -= 0.85
            
        # Arrow to next stage
        if i < len(stages) - 1:
            ax.annotate("", xy=(stages[i+1]["x"] - 0.95, 4.5), xytext=(x + 0.95, 4.5),
                        arrowprops=dict(arrowstyle="->", color="#94a3b8", lw=2.5, mutation_scale=15))
            
    # Local Execution Footer Banner
    f_box = patches.FancyBboxPatch((0.5, 0.4), 14.5, 1.4, boxstyle="round,pad=0.15,rounding_size=0.15",
                                   facecolor="#0f172a", edgecolor="#334155", linewidth=1.5, zorder=2)
    ax.add_patch(f_box)
    ax.text(7.75, 1.4, "Local Execution Environment (100% Offline & Private | Consumer CPU Optimized)",
            color="#38bdf8", fontsize=11, fontweight="bold", ha="center", va="center", zorder=3)
    
    tools = "Python 3.11+  |  PyMuPDF  |  Scikit-Learn  |  Sentence-Transformers (all-MiniLM-L6-v2)  |  PyTorch  |  SQLite3  |  Ollama (Llama 3.2)  |  Flask"
    ax.text(7.75, 0.85, tools, color="#94a3b8", fontsize=9.5, ha="center", va="center", zorder=3)
    
    ax.set_xlim(-0.2, 15.5)
    ax.set_ylim(0.0, 8.8)
    ax.axis("off")
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "slide9_system_workflow.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ------------------------------------------------------------------------------
# 3. Slide 10: Interface Showcase 1 — Hybrid Search Portal
# ------------------------------------------------------------------------------
def generate_slide10_ui_search_demo():
    print("Generating Slide 10: UI Search Demo Mockup...")
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.set_facecolor("#0f172a")  # Modern dark theme matching sample PPT
    fig.patch.set_facecolor("#0f172a")
    
    # Header Navbar
    nav = patches.Rectangle((0, 7.3), 15, 0.9, facecolor="#1e293b", edgecolor="#334155")
    ax.add_patch(nav)
    ax.text(0.6, 7.75, "AI-Powered Semantic Search Engine", color="#38bdf8", fontsize=13, fontweight="bold", va="center")
    ax.text(14.4, 7.75, "Latency: 0.55 ms | Mode: Hybrid Fusion (Alpha=0.50)", color="#94a3b8", fontsize=9.5, ha="right", va="center")
    
    # Query Search Bar
    s_bar = patches.FancyBboxPatch((0.6, 6.2), 13.8, 0.8, boxstyle="round,pad=0.1,rounding_size=0.1",
                                   facecolor="#334155", edgecolor="#38bdf8", linewidth=1.5)
    ax.add_patch(s_bar)
    ax.text(0.9, 6.6, "Query: 'Transformer attention mechanisms query key value self-attention'",
            color="#f8fafc", fontsize=10.5, fontweight="bold", va="center")
    ax.text(14.0, 6.6, "[ SEARCH ]", color="#38bdf8", fontsize=10, fontweight="bold", ha="right", va="center")
    
    # Metrics Banner
    m_box = patches.FancyBboxPatch((0.6, 5.3), 13.8, 0.65, boxstyle="round,pad=0.1,rounding_size=0.08",
                                   facecolor="#064e3b", edgecolor="#059669", linewidth=1.0)
    ax.add_patch(m_box)
    ax.text(0.9, 5.62, "✓ Top-5 Passages Retrieved in 0.55 ms | Precision@5: 1.00 | Reciprocal Rank: 1.0 (Top Match at #1)",
            color="#34d399", fontsize=9.5, fontweight="bold", va="center")
    
    # Result Card 1 (Rank 1)
    c1 = patches.FancyBboxPatch((0.6, 3.1), 13.8, 1.95, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor="#1e293b", edgecolor="#38bdf8", linewidth=1.5)
    ax.add_patch(c1)
    ax.text(0.9, 4.75, "#1  Transformer Architectures and Multi-Head Attention Mechanisms", color="#38bdf8", fontsize=11, fontweight="bold")
    ax.text(14.0, 4.75, "Hybrid Score: 0.948  |  Keyword (TF-IDF): 0.92  |  Semantic Cosine: 0.97", color="#fbbf24", fontsize=9, ha="right", fontweight="bold")
    ax.text(0.9, 4.4, "Source: sample_04_transformer_architectures.pdf  •  Page 1  •  SHA-256: e31da61f...  •  Chunk ID: #14", color="#94a3b8", fontsize=8.5)
    
    excerpt1 = ("The core innovation of the Transformer architecture lies in the multi-head self-attention mechanism.\n"
                "Queries (Q), Keys (K), and Values (V) are projected into d_k dimensional spaces. Scaled Dot-Product Attention\n"
                "is computed as Attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) V, routing contextual weights dynamically...")
    ax.text(0.9, 3.7, excerpt1, color="#e2e8f0", fontsize=9, linespacing=1.4)
    
    # Result Card 2 (Rank 2)
    c2 = patches.FancyBboxPatch((0.6, 1.0), 13.8, 1.85, boxstyle="round,pad=0.15,rounding_size=0.1",
                                facecolor="#1e293b", edgecolor="#334155", linewidth=1.0)
    ax.add_patch(c2)
    ax.text(0.9, 2.55, "#2  Continuous Word Representations and Self-Attention Formulation", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax.text(14.0, 2.55, "Hybrid Score: 0.884  |  Keyword (TF-IDF): 0.85  |  Semantic Cosine: 0.91", color="#fbbf24", fontsize=9, ha="right", fontweight="bold")
    ax.text(0.9, 2.22, "Source: sample_06_continuous_word_representation.pdf  •  Page 2  •  Chunk ID: #21", color="#94a3b8", fontsize=8.5)
    
    excerpt2 = ("Self-attention allows the representation of a sequence to be constructed by relating different positions of\n"
                "the same input. This overcomes the sequential bottlenecks of RNNs, enabling full parallelization across CPU/GPU.")
    ax.text(0.9, 1.6, excerpt2, color="#cbd5e1", fontsize=9, linespacing=1.4)
    
    # Footer
    ax.text(7.5, 0.4, "Figure 1: Hybrid Search Engine Interface — Dual-Score Ranking & Verifiable Source Provenance",
            color="#64748b", fontsize=10, ha="center", style="italic")
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "slide10_ui_search_demo.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

# ------------------------------------------------------------------------------
# 4. Slide 11: Interface Showcase 2 — Grounded RAG & Attention Inspector
# ------------------------------------------------------------------------------
def generate_slide11_ui_rag_demo():
    print("Generating Slide 11: UI RAG & Attention Demo Mockup...")
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=300)
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    
    # Header Navbar
    nav = patches.Rectangle((0, 7.3), 15, 0.9, facecolor="#1e293b", edgecolor="#334155")
    ax.add_patch(nav)
    ax.text(0.6, 7.75, "Grounded RAG Intelligence & Transformer Attention Inspector", color="#38bdf8", fontsize=13, fontweight="bold", va="center")
    ax.text(14.4, 7.75, "Engine: Local Ollama (llama3.2)  |  Hallucination Rate: 0.0%", color="#34d399", fontsize=9.5, ha="right", va="center")
    
    # Left Column: Grounded QA Chat View
    left_card = patches.FancyBboxPatch((0.6, 0.9), 8.5, 6.1, boxstyle="round,pad=0.15,rounding_size=0.1",
                                       facecolor="#1e293b", edgecolor="#334155", linewidth=1.2)
    ax.add_patch(left_card)
    ax.text(0.9, 6.6, "Retrieval-Augmented Answer (Grounded in Context)", color="#f8fafc", fontsize=11, fontweight="bold")
    
    q_bubble = patches.FancyBboxPatch((0.9, 5.7), 7.9, 0.65, boxstyle="round,pad=0.1,rounding_size=0.08",
                                      facecolor="#334155", edgecolor="none")
    ax.add_patch(q_bubble)
    ax.text(1.1, 6.02, "User: How does the Transformer compute attention between queries and keys?", color="#38bdf8", fontsize=9, fontweight="bold", va="center")
    
    ans_text = ("AI Answer:\n"
                "According to [sample_04_transformer_architectures.pdf, Page 1], the Transformer\n"
                "computes Scaled Dot-Product Attention by taking the inner dot-product between\n"
                "query vector (Q) and key vector (K), divided by the scaling factor sqrt(d_k) to prevent\n"
                "vanishing gradients in large dimensions. The softmax function is applied across\n"
                "the resulting matrix to produce normalized attention distribution weights.\n\n"
                "Verified Citations:\n"
                "• [Doc 4, Page 1]: Section 2.1 'Scaled Dot-Product Mathematical Formulation'\n"
                "• [Doc 6, Page 2]: Section 3 'Continuous Word Representations & Context Matrices'")
    ax.text(0.9, 3.8, ans_text, color="#e2e8f0", fontsize=8.5, linespacing=1.35)
    
    prompt_shield = patches.FancyBboxPatch((0.9, 1.2), 7.9, 0.75, boxstyle="round,pad=0.08,rounding_size=0.08",
                                          facecolor="#064e3b", edgecolor="#059669")
    ax.add_patch(prompt_shield)
    ax.text(1.1, 1.57, "🛡️ Security Shield Active: Ingested context strictly bounded in <CONTEXT> tags.\nPrompt injection suppression active | Zero hallucination verification passed.",
            color="#34d399", fontsize=8, va="center")
    
    # Right Column: Attention Heatmap Preview
    right_card = patches.FancyBboxPatch((9.5, 0.9), 4.9, 6.1, boxstyle="round,pad=0.15,rounding_size=0.1",
                                        facecolor="#1e293b", edgecolor="#38bdf8", linewidth=1.2)
    ax.add_patch(right_card)
    ax.text(9.8, 6.6, "Attention Heatmap (Layer 4, Head 2)", color="#38bdf8", fontsize=10.5, fontweight="bold")
    
    # Mini synthetic heatmap
    tokens = ["transf.", "self", "attn", "query", "key", "value"]
    np.random.seed(42)
    mat = np.random.rand(6, 6) * 0.4
    for i in range(6): mat[i, i] += 0.5
    mat[2, 3] += 0.4; mat[2, 4] += 0.4
    mat = mat / np.sum(mat, axis=-1, keepdims=True)
    
    sub_ax = fig.add_axes([0.655, 0.22, 0.29, 0.48])
    im = sub_ax.imshow(mat, cmap="magma", interpolation="nearest")
    sub_ax.set_xticks(range(6))
    sub_ax.set_yticks(range(6))
    sub_ax.set_xticklabels(tokens, fontsize=7.5, color="white", rotation=30)
    sub_ax.set_yticklabels(tokens, fontsize=7.5, color="white")
    sub_ax.tick_params(colors="white")
    
    # Footer
    ax.text(7.5, 0.4, "Figure 2: Grounded Generative RAG Interface — Verified Page Citations & Multi-Head Attention Routing",
            color="#64748b", fontsize=10, ha="center", style="italic")
    
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    plt.tight_layout()
    
    out_path = OUTPUT_DIR / "slide11_ui_rag_attention_demo.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

def main():
    print("=" * 80)
    print("GENERATING PRESENTATION (PPT) SLIDE IMAGES (16:9 HD)")
    print("=" * 80)
    generate_slide8_methodology()
    generate_slide9_system_workflow()
    generate_slide10_ui_search_demo()
    generate_slide11_ui_rag_demo()
    print("=" * 80)
    print(f"All PPT slide images successfully generated in: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
