"""Unit tests for NLP Preprocessor module."""
import pytest
from services.preprocessor import (
    normalize_text,
    tokenize_text,
    remove_stopwords,
    apply_stemming,
    apply_lemmatization,
    preprocess_for_tfidf,
    preprocess_for_embeddings,
    explain_preprocessing_pipeline
)

def test_normalize_text():
    text = "  Hello   WORLD! 123 \t\n  "
    norm = normalize_text(text, lowercase=True, remove_punctuation=True, remove_numbers=True)
    assert "hello" in norm
    assert "world" in norm
    assert "123" not in norm
    assert "!" not in norm

def test_tokenize_text():
    text = "Machine learning and deep-learning algorithms detect diseases."
    tokens = tokenize_text(text)
    assert "Machine" in tokens or "machine" in [t.lower() for t in tokens]
    assert "deep-learning" in tokens
    assert len(tokens) >= 6

def test_remove_stopwords():
    tokens = ["the", "quick", "brown", "fox", "is", "over", "there"]
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "is" not in filtered
    assert "fox" in filtered
    assert "quick" in filtered

def test_apply_stemming():
    tokens = ["operating", "operations", "operator", "university"]
    stemmed = apply_stemming(tokens)
    assert len(stemmed) == 4
    assert stemmed[0].startswith("oper")

def test_apply_lemmatization():
    tokens = ["diseases", "algorithms", "better"]
    lemmatized = apply_lemmatization(tokens)
    assert "disease" in lemmatized or "diseases" in lemmatized or "diseas" in lemmatized
    assert len(lemmatized) == 3

def test_preprocess_for_tfidf():
    text = "The artificial intelligence algorithms are detecting diseases early."
    clean = preprocess_for_tfidf(text)
    assert "the" not in clean.split()
    assert "are" not in clean.split()
    assert len(clean.split()) >= 3

def test_preprocess_for_embeddings():
    text = "How can AI detect diseases early? Machine learning helps doctors."
    res = preprocess_for_embeddings(text)
    # Casing and punctuation should be preserved
    assert "?" in res
    assert "AI" in res
    assert "Machine" in res

def test_explain_preprocessing_pipeline():
    text = "The doctor is diagnosing complex chronic diseases."
    explanation = explain_preprocessing_pipeline(text)
    assert "original_text" in explanation
    assert "normalized_text" in explanation
    assert "tokens" in explanation
    assert "stopwords_removed" in explanation
    assert "statistics" in explanation
    assert explanation["statistics"]["reduction_percentage"] > 0
