"""NLP Visualization Service: Word Cloud, Bigram Network Graph, and Frequency Bar Chart.
Supports corpus-level and document-level visualizations rendered via WordCloud, NetworkX, and Matplotlib.
"""
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
from collections import Counter
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud

from services.preprocessor import normalize_text, tokenize_text, remove_stopwords, apply_lemmatization
from database import operations

class VisualizationService:
    """Generates Word Clouds, Bigram Co-occurrence Networks, and Frequency Bar Charts."""
    
    def __init__(self):
        pass

    def extract_cleaned_tokens(self, text: str) -> List[str]:
        """Normalize, tokenize, and remove stopwords for human-readable linguistic visualizations."""
        norm = normalize_text(text, lowercase=True, remove_punctuation=True, remove_numbers=True)
        tokens = tokenize_text(norm)
        filtered = remove_stopwords(tokens)
        # Keep meaningful words with length >= 3
        return [t.lower() for t in filtered if len(t) >= 3]

    def extract_bigrams(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """Extract consecutive adjacent word pairs (bigrams)."""
        bigrams = []
        for i in range(len(tokens) - 1):
            w1, w2 = tokens[i], tokens[i + 1]
            if w1 != w2:  # Avoid self-loops
                bigrams.append((w1, w2))
        return bigrams

    def generate_word_cloud(self, text: str, max_words: int = 80) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a Word Cloud image encoded as base64 PNG and return top word frequencies."""
        tokens = self.extract_cleaned_tokens(text)
        if not tokens:
            tokens = ["search", "engine", "natural", "language", "processing", "information", "retrieval"]
            
        freq_map = Counter(tokens)
        top_words = [{"word": w, "count": c} for w, c in freq_map.most_common(20)]
        
        # Generate WordCloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
            max_words=max_words,
            contour_width=1,
            contour_color="#e2e8f0",
            random_state=42
        ).generate_from_frequencies(freq_map)
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        return base64_img, top_words

    def generate_bigram_network(self, text: str, top_n: int = 25) -> Tuple[str, List[Dict[str, Any]]]:
        """Build and render a Bigram Co-occurrence Network Graph using NetworkX and Matplotlib."""
        tokens = self.extract_cleaned_tokens(text)
        bigrams = self.extract_bigrams(tokens)
        
        if not bigrams:
            bigrams = [
                ("machine", "learning"), ("deep", "learning"), ("neural", "network"),
                ("natural", "language"), ("search", "engine"), ("information", "retrieval"),
                ("early", "detection"), ("disease", "detection")
            ]
            
        bigram_counts = Counter(bigrams)
        top_bigrams = bigram_counts.most_common(top_n)
        
        # Build NetworkX Graph
        G = nx.Graph()
        for (w1, w2), weight in top_bigrams:
            G.add_edge(w1, w2, weight=weight)
            
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        # Spring layout with fixed seed for determinism
        pos = nx.spring_layout(G, k=0.9, iterations=50, seed=42)
        
        # Node degrees and sizes
        node_degrees = dict(G.degree())
        node_sizes = [max(600, node_degrees[n] * 450) for n in G.nodes()]
        
        # Draw edges with varying thickness based on co-occurrence weight
        edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(edge_weights) if edge_weights else 1
        edge_widths = [1.2 + (w / max_w) * 3.5 for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color="#94a3b8", alpha=0.7)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color="#3b82f6", alpha=0.9)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_family="sans-serif", font_weight="bold", font_color="#0f172a")
        
        # Edge labels for highest frequency transitions
        edge_labels = {(u, v): str(d["weight"]) for u, v, d in G.edges(data=True) if d["weight"] > 1}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="#ef4444", ax=ax)
        
        ax.axis("off")
        plt.title("Bigram Co-occurrence Linguistic Network Graph", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        bigram_list = [{"bigram": f"{w1} → {w2}", "count": count} for (w1, w2), count in top_bigrams]
        return base64_img, bigram_list

    def generate_barchart(self, text: str, chart_type: str = "unigram", top_k: int = 15) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate a Term Frequency Bar Chart using Matplotlib."""
        tokens = self.extract_cleaned_tokens(text)
        
        if chart_type == "bigram":
            bigrams = self.extract_bigrams(tokens)
            counts = Counter([f"{w1} {w2}" for w1, w2 in bigrams]).most_common(top_k)
            title = f"Top {top_k} Most Frequent Bigrams (Word Pairs)"
            color = "#10b981"  # Emerald green
        else:
            counts = Counter(tokens).most_common(top_k)
            title = f"Top {top_k} Most Frequent Vocabulary Terms (Unigrams)"
            color = "#2563eb"  # Royal blue
            
        if not counts:
            counts = [("sample", 5), ("terms", 3), ("data", 2)]
            
        # Reverse for horizontal bar chart (highest at top)
        labels = [item[0] for item in reversed(counts)]
        values = [item[1] for item in reversed(counts)]
        
        fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
        bars = ax.barh(labels, values, color=color, alpha=0.85, edgecolor="#1e293b", linewidth=0.5)
        
        # Add frequency values at end of each bar
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{int(width)}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#1e293b"
            )
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.set_xlabel("Occurrence Frequency (Count)", fontsize=10, labelpad=8)
        plt.title(title, fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        plt.close(fig)
        buf.seek(0)
        base64_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        data_list = [{"label": l, "count": c} for l, c in counts]
        return base64_img, data_list
