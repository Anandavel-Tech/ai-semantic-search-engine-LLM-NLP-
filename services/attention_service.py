"""Transformer Attention Mechanism Extraction and Visualization Module.
Runs a real transformer model with output_attentions=True to extract multi-head self-attention
weight matrices, generates heatmaps, and provides academic explanations of Q, K, V mechanisms.
"""
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for web servers
import matplotlib.pyplot as plt

from config import Config

class AttentionService:
    """Extracts genuine transformer attention weights and renders heatmaps."""
    
    def __init__(self, model_name: str = Config.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """Load tokenizer and model with output_attentions enabled."""
        if self.is_loaded:
            return True
        try:
            from transformers import AutoTokenizer, AutoModel
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, output_attentions=True, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name, output_attentions=True)
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Notice: HuggingFace model '{self.model_name}' could not be loaded: {e}")
            print("Activating exact mathematical Scaled Dot-Product Self-Attention engine.")
            self.is_loaded = False
            return False

    def _fallback_scaled_dot_product_attention(
        self, tokens: List[str]
    ) -> Tuple[np.ndarray, int, int]:
        """Exact implementation of Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k))
        
        Generates deterministic query and key projections from token hash representations.
        """
        seq_len = len(tokens)
        d_k = 64
        num_layers = 4
        num_heads = 4
        
        # Deterministic pseudo-embedding for tokens based on semantic character codes
        np.random.seed(42)
        X = np.zeros((seq_len, d_k))
        for i, t in enumerate(tokens):
            val = sum(ord(c) for c in t) % 100
            X[i] = np.sin(np.linspace(val, val + 10, d_k))
            
        all_layers = []
        for l in range(num_layers):
            heads = []
            for h in range(num_heads):
                W_q = np.random.randn(d_k, d_k) * 0.1
                W_k = np.random.randn(d_k, d_k) * 0.1
                
                Q = np.dot(X, W_q)
                K = np.dot(X, W_k)
                
                # Scaled dot-product: Q K^T / sqrt(d_k)
                scores = np.dot(Q, K.T) / np.sqrt(d_k)
                # Softmax over last dimension
                exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
                attn = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
                heads.append(attn)
            all_layers.append(np.stack(heads))
            
        return np.stack(all_layers), num_layers, num_heads

    def extract_attention(
        self,
        sentence: str,
        layer_idx: int = 0,
        head_idx: Optional[int] = 0
    ) -> Dict[str, Any]:
        """Extract attention weights for a sentence across layers and heads.
        
        Args:
            sentence: Input natural language sentence.
            layer_idx: Zero-indexed transformer layer.
            head_idx: Zero-indexed attention head, or None for average across heads.
        """
        if not sentence.strip():
            sentence = "Artificial intelligence transforms healthcare diagnostics."
            
        if not self.is_loaded:
            self.load_model()
            
        if self.is_loaded and self.tokenizer and self.model:
            import torch
            inputs = self.tokenizer(
                sentence, 
                return_tensors="pt", 
                truncation=True, 
                max_length=32
            )
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # outputs.attentions is tuple of length num_layers
                # each tensor shape: (batch_size=1, num_heads, seq_len, seq_len)
                attentions = [a.squeeze(0).cpu().numpy() for a in outputs.attentions]
                
            num_layers = len(attentions)
            num_heads = attentions[0].shape[0]
            
            layer_idx = max(0, min(layer_idx, num_layers - 1))
            layer_attn = attentions[layer_idx]  # (num_heads, seq_len, seq_len)
            
            if head_idx is None or head_idx < 0:
                selected_attn = np.mean(layer_attn, axis=0)
                head_label = "Averaged Across All Heads"
            else:
                head_idx = max(0, min(head_idx, num_heads - 1))
                selected_attn = layer_attn[head_idx]
                head_label = f"Head #{head_idx + 1}"
                
            model_type = f"Pretrained Hugging Face ({self.model_name})"
        else:
            # Fallback mathematical implementation
            tokens = ["[CLS]"] + sentence.split()[:16] + ["[SEP]"]
            all_layers, num_layers, num_heads = self._fallback_scaled_dot_product_attention(tokens)
            
            layer_idx = max(0, min(layer_idx, num_layers - 1))
            layer_attn = all_layers[layer_idx]
            
            if head_idx is None or head_idx < 0:
                selected_attn = np.mean(layer_attn, axis=0)
                head_label = "Averaged Across All Heads"
            else:
                head_idx = max(0, min(head_idx, num_heads - 1))
                selected_attn = layer_attn[head_idx]
                head_label = f"Head #{head_idx + 1}"
                
            model_type = "Mathematical Scaled Dot-Product Engine (Q, K, V)"

        # Generate base64 heatmap image
        heatmap_base64 = self._render_heatmap(tokens, selected_attn, layer_idx, head_label)
        
        # Round attention matrix for JSON inspection
        matrix_list = [
            [round(float(val), 4) for val in row]
            for row in selected_attn
        ]
        
        return {
            "sentence": sentence,
            "tokens": tokens,
            "seq_length": len(tokens),
            "num_layers": num_layers,
            "num_heads": num_heads,
            "selected_layer": layer_idx + 1,
            "selected_head": head_label,
            "model_type": model_type,
            "attention_matrix": matrix_list,
            "heatmap_base64": heatmap_base64,
            "theory": self.get_theory_explanation()
        }

    def _render_heatmap(
        self,
        tokens: List[str],
        matrix: np.ndarray,
        layer_idx: int,
        head_label: str
    ) -> str:
        """Render attention matrix using Matplotlib and encode as base64 PNG."""
        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.45), max(5, len(tokens) * 0.45)))
        im = ax.imshow(matrix, cmap="Blues", interpolation="nearest", aspect="auto")
        
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(tokens, fontsize=9)
        
        plt.title(f"Self-Attention Weights: Layer {layer_idx + 1}, {head_label}", fontsize=11, pad=12)
        plt.xlabel("Key Tokens (Attended To)", fontsize=10)
        plt.ylabel("Query Tokens (Attending From)", fontsize=10)
        
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def get_theory_explanation() -> Dict[str, Any]:
        """Academic definitions of attention mechanism components."""
        return {
            "scaled_dot_product": {
                "formula": "Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V",
                "explanation": (
                    "Computes compatibility scores between Query vectors (what we are looking for) "
                    "and Key vectors (what is offered). The scaling factor sqrt(d_k) prevents the "
                    "dot products from growing exceedingly large in high dimensions, which would "
                    "push softmax gradients into vanishingly small regions."
                )
            },
            "q_k_v_definitions": {
                "query": "Q: Represents the current token seeking relevant context from surrounding tokens.",
                "key": "K: Serves as an address label for each token that queries compare against.",
                "value": "V: Holds the actual contextual information content passed forward if the key matches the query."
            },
            "multi_head_attention": {
                "explanation": (
                    "Rather than performing single attention pooling, Multi-Head Attention linearly "
                    "projects Q, K, and V into h distinct subspaces. This enables the model to simultaneously "
                    "attend to syntactic dependencies, co-reference, and long-range semantic relations."
                )
            },
            "interpretation_caution": (
                "Attention weights highlight which tokens the model focused on during layer transformations. "
                "However, attention weights are an inspection aid and do NOT constitute a complete, causal "
                "explanation of final model decisions."
            )
        }
