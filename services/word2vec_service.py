"""Educational Word2Vec Module using Gensim.
Demonstrates Continuous Bag of Words (CBOW) and Skip-gram architectures,
vocabulary construction, vector arithmetic, and similarity calculation.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from gensim.models import Word2Vec
from services.preprocessor import normalize_text, tokenize_text, remove_stopwords, apply_lemmatization

class Word2VecService:
    """Trains and queries Gensim Word2Vec models for educational exploration."""
    
    def __init__(self, vector_size: int = 50, window: int = 5, min_count: int = 1):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.cbow_model: Optional[Word2Vec] = None
        self.skipgram_model: Optional[Word2Vec] = None
        self.is_trained = False
        
    def prepare_sentences(self, texts: List[str]) -> List[List[str]]:
        """Tokenize and clean raw documents into lists of word tokens for training."""
        sentences = []
        for text in texts:
            norm = normalize_text(text, lowercase=True, remove_punctuation=True)
            tokens = tokenize_text(norm)
            clean_tokens = remove_stopwords(tokens)
            if clean_tokens:
                sentences.append(clean_tokens)
        return sentences

    def train(self, texts: List[str], epochs: int = 30) -> Dict[str, Any]:
        """Train both CBOW (sg=0) and Skip-gram (sg=1) models on the corpus."""
        sentences = self.prepare_sentences(texts)
        if not sentences:
            return {"status": "error", "message": "No valid tokens found to train Word2Vec."}
            
        # Train CBOW (Continuous Bag-of-Words: predicts center word from context)
        self.cbow_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=0,  # 0 for CBOW
            epochs=epochs,
            workers=1,
            seed=42
        )
        
        # Train Skip-gram (predicts context words given center word)
        self.skipgram_model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            sg=1,  # 1 for Skip-gram
            epochs=epochs,
            workers=1,
            seed=42
        )
        
        self.is_trained = True
        vocab_size = len(self.cbow_model.wv.key_to_index)
        
        return {
            "status": "success",
            "vocabulary_size": vocab_size,
            "sentence_count": len(sentences),
            "vector_dimension": self.vector_size,
            "epochs": epochs
        }

    def get_model(self, architecture: str = "cbow") -> Optional[Word2Vec]:
        """Retrieve model instance based on architecture parameter."""
        if architecture.lower() == "skipgram":
            return self.skipgram_model
        return self.cbow_model

    def get_word_vector(self, word: str, architecture: str = "cbow") -> Optional[List[float]]:
        """Retrieve the dense embedding vector for a given word."""
        model = self.get_model(architecture)
        if not model or word.lower() not in model.wv:
            return None
        return [round(float(val), 4) for val in model.wv[word.lower()]]

    def find_most_similar(
        self,
        positive: List[str],
        negative: Optional[List[str]] = None,
        top_n: int = 5,
        architecture: str = "cbow"
    ) -> List[Dict[str, Any]]:
        """Find most similar words using cosine similarity or vector arithmetic."""
        model = self.get_model(architecture)
        if not model:
            return []
            
        pos = [w.lower() for w in positive if w.lower() in model.wv]
        neg = [w.lower() for w in (negative or []) if w.lower() in model.wv]
        
        if not pos:
            return []
            
        try:
            similar_pairs = model.wv.most_similar(positive=pos, negative=neg, topn=top_n)
            return [
                {"word": word, "similarity": round(float(sim), 4)}
                for word, sim in similar_pairs
            ]
        except Exception:
            return []

    def compute_similarity(self, word1: str, word2: str, architecture: str = "cbow") -> Optional[float]:
        """Calculate cosine similarity between two words."""
        model = self.get_model(architecture)
        if not model:
            return None
        w1, w2 = word1.lower(), word2.lower()
        if w1 not in model.wv or w2 not in model.wv:
            return None
        return round(float(model.wv.similarity(w1, w2)), 4)

    def get_vocabulary(self, architecture: str = "cbow", limit: int = 100) -> List[str]:
        """Return list of indexed vocabulary tokens."""
        model = self.get_model(architecture)
        if not model:
            return []
        return list(model.wv.key_to_index.keys())[:limit]

    @staticmethod
    def get_educational_overview() -> Dict[str, Any]:
        """Provide detailed educational notes on CBOW vs. Skip-gram."""
        return {
            "cbow": {
                "name": "Continuous Bag-of-Words (CBOW)",
                "mechanism": "Predicts the target center word given the average representation of context window words.",
                "formula": "P(w_t | w_{t-c}, ..., w_{t+c}) = softmax(v'_{w_t}^T \\bar{v}_c)",
                "strengths": "Fast training, smooth representations, performs well on frequent words.",
                "best_for": "Larger corpora, faster execution."
            },
            "skipgram": {
                "name": "Continuous Skip-gram",
                "mechanism": "Predicts multiple context words surrounding a given input center word.",
                "formula": "\\sum_{-c \\le j \\le c, j \\ne 0} \\log P(w_{t+j} | w_t)",
                "strengths": "Works well with rare words and smaller datasets because each word produces multiple training pairs.",
                "best_for": "Capturing subtle semantic relationships and uncommon terms."
            },
            "limitations_vs_transformers": {
                "static_embeddings": "Word2Vec assigns a single fixed vector per token, failing to disambiguate polysemy (e.g. 'apple' fruit vs. 'apple' tech company).",
                "contextual_transformers": "Models like BERT and Sentence-Transformers use self-attention to generate dynamic, contextual embeddings that vary depending on sentence context."
            }
        }
