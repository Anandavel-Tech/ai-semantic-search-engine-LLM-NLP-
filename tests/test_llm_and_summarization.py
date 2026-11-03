"""Unit tests for LLM RAG Pipeline, Citations, and Summarization."""
import pytest
from services.llm_service import LLMService
from services.summarization_service import SummarizationService

@pytest.fixture
def mock_retrieved_chunks():
    return [
        {
            "chunk_id": 1,
            "document_id": 101,
            "doc_title": "Chronic Kidney Disease ML",
            "page_number": 1,
            "text": "Machine learning classifiers such as Random Forest achieve high sensitivity in detecting early chronic kidney disease from routine blood tests.",
            "score": 0.88
        },
        {
            "chunk_id": 2,
            "document_id": 102,
            "doc_title": "Diabetic Retinopathy Screening",
            "page_number": 2,
            "text": "Deep learning models analyze fundus photographs to grade diabetic retinopathy severity and prevent blindness.",
            "score": 0.75
        }
    ]

def test_rag_extractive_fallback_qa(mock_retrieved_chunks):
    llm = LLMService()
    # Force fallback provider
    llm.provider = "extractive_fallback"
    
    question = "How does machine learning detect early chronic kidney disease?"
    result = llm.generate_rag_answer(question, mock_retrieved_chunks)

    assert "answer" in result
    assert result["has_sufficient_evidence"] is True
    assert "Chronic Kidney Disease ML" in result["answer"] or len(result["citations"]) > 0
    assert len(result["citations"]) >= 1
    assert result["citations"][0]["doc_title"] == "Chronic Kidney Disease ML"

def test_rag_insufficient_evidence(mock_retrieved_chunks):
    llm = LLMService()
    llm.provider = "extractive_fallback"
    
    # Query completely unrelated to the retrieved chunks
    unrelated_question = "What is quantum entanglement in theoretical physics?"
    result = llm.generate_rag_answer(unrelated_question, mock_retrieved_chunks)

    assert result["has_sufficient_evidence"] is False
    assert "insufficient evidence" in result["answer"].lower()

def test_single_document_summarization():
    summarizer = SummarizationService()
    long_text = (
        "Chronic kidney disease affects millions of patients worldwide. "
        "Early symptoms are frequently mild and overlooked by patients. "
        "Machine learning models analyze routine blood panels to predict renal decline. "
        "Random Forest and XGBoost achieve high diagnostic accuracy on test datasets. "
        "Early dietary intervention delays disease progression by several years. "
        "Clinical decision support systems alert primary care physicians promptly."
    )
    res = summarizer.summarize_document(long_text, title="Kidney Health Study")
    assert "summary" in res
    assert len(res["summary"]) > 20
    assert "method" in res

def test_search_results_summarization(mock_retrieved_chunks):
    summarizer = SummarizationService()
    res = summarizer.summarize_search_results("How can AI assist healthcare?", mock_retrieved_chunks)
    assert "summary" in res
    assert len(res["sources"]) >= 1
