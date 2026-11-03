"""Document and Search-Result Summarization Module.
Provides Single-Document, Multi-Document, and Search-Result summarization
using LLM prompting with an intelligent extractive TextRank fallback.
"""
from typing import List, Dict, Any, Optional
import re
from config import Config
from services.preprocessor import tokenize_text, normalize_text
from services.llm_service import LLMService

class SummarizationService:
    """Multi-strategy summarizer with LLM synthesis and extractive fallback."""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm = llm_service or LLMService()

    def _extractive_summarize(self, text: str, max_sentences: int = 4) -> str:
        """Heuristic sentence-ranking summarizer based on keyword density and lead bias."""
        if not text:
            return "No text available to summarize."
            
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip().split()) >= 5]
        if not sentences:
            return text[:300]
            
        if len(sentences) <= max_sentences:
            return " ".join(sentences)
            
        # Frequency table of informative tokens
        words = tokenize_text(normalize_text(text, lowercase=True, remove_punctuation=True))
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
            
        scored = []
        for idx, s in enumerate(sentences):
            s_words = tokenize_text(normalize_text(s, lowercase=True))
            score = sum(freq.get(w, 0) for w in s_words) / max(len(s_words), 1)
            # Add lead bias bonus for introductory sentences
            if idx < 3:
                score *= 1.25
            scored.append((score, idx, s))
            
        # Select top scoring sentences and restore original order
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_sentences]
        top.sort(key=lambda x: x[1])
        
        return " ".join(s for _, _, s in top)

    def summarize_document(self, text: str, title: str = "Document") -> Dict[str, Any]:
        """Generate a concise executive summary of a single document."""
        # Check LLM availability
        prompt = (
            f"Provide a professional 3-to-4 sentence executive summary of the following document:\n\n"
            f"Title: {title}\n"
            f"Content:\n{text[:3000]}\n\n"
            f"Summary:"
        )
        
        raw = None
        if self.llm.provider == "ollama":
            raw = self.llm._call_ollama(prompt)
        elif self.llm.provider == "openai_compatible":
            raw = self.llm._call_openai(prompt)
            
        if raw and len(raw) > 30:
            return {
                "summary": raw,
                "method": f"LLM ({self.llm.provider})",
                "title": title
            }
            
        # Fallback
        summary = self._extractive_summarize(text, max_sentences=4)
        return {
            "summary": summary,
            "method": "Extractive NLP Centroid",
            "title": title
        }

    def summarize_search_results(self, query: str, search_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize retrieved passages into a unified summary answering the query."""
        if not search_results:
            return {
                "summary": "No search results to synthesize.",
                "method": "None",
                "sources": []
            }
            
        combined_text = "\n\n".join(
            f"[{r.get('doc_title')}, Page {r.get('page_number', 1)}]: {r.get('text', '')}"
            for r in search_results[:4]
        )
        
        prompt = (
            f"Synthesize the following search results for the user query: '{query}'.\n"
            f"Highlight key consensus points, cite document names, and maintain brevity (3 sentences).\n\n"
            f"Passages:\n{combined_text}\n\n"
            f"Synthesis:"
        )
        
        raw = None
        if self.llm.provider == "ollama":
            raw = self.llm._call_ollama(prompt)
        elif self.llm.provider == "openai_compatible":
            raw = self.llm._call_openai(prompt)
            
        if raw and len(raw) > 30:
            return {
                "summary": raw,
                "method": f"LLM ({self.llm.provider})",
                "sources": [r.get("doc_title") for r in search_results[:4]]
            }
            
        summary = self._extractive_summarize(combined_text, max_sentences=3)
        return {
            "summary": summary,
            "method": "Extractive Multi-Passage Synthesis",
            "sources": [r.get("doc_title") for r in search_results[:4]]
        }
