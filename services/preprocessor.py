"""NLP Preprocessing Module.
Implements text normalization, tokenization, stopword removal,
stemming, lemmatization, and specialized pipelines for TF-IDF vs. Embeddings.
"""
import re
import string
import unicodedata
from typing import List, Dict, Any, Optional

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Standard English stop words fallback in case NLTK corpus is offline
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", 
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", 
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", 
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", 
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", 
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", 
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", 
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", 
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", 
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", 
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", 
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", 
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", 
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", 
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", 
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}

# Initialize NLTK components gracefully
try:
    nltk.data.find("corpora/stopwords")
    from nltk.corpus import stopwords
    STOPWORDS_SET = set(stopwords.words("english"))
except Exception:
    STOPWORDS_SET = DEFAULT_STOPWORDS

stemmer = PorterStemmer()
try:
    nltk.data.find("corpora/wordnet")
    lemmatizer = WordNetLemmatizer()
except Exception:
    lemmatizer = None

def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = False,
    remove_numbers: bool = False
) -> str:
    """Normalize text through unicode decomposition, case folding, and regex cleaning.
    
    Args:
        text: Input string.
        lowercase: Whether to convert text to lower case.
        remove_punctuation: Whether to strip punctuation characters.
        remove_numbers: Whether to strip numerical digits.
        
    Returns:
        Cleaned, normalized string.
    """
    if not text:
        return ""
        
    # Unicode NFKD normalization (converts accents, ligatures)
    text = unicodedata.normalize("NFKD", text)
    
    if lowercase:
        text = text.lower()
        
    if remove_punctuation:
        text = text.translate(str.maketrans("", "", string.punctuation))
        
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
        
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text: str) -> List[str]:
    """Tokenize string into individual word tokens.
    
    Uses regex alphanumeric tokenization to ensure stability without external models.
    """
    if not text:
        return []
    # Match alphanumeric sequences including hyphens for compound words
    tokens = re.findall(r"\b[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*\b", text)
    return tokens

def remove_stopwords(tokens: List[str], custom_stopwords: Optional[set] = None) -> List[str]:
    """Filter out non-informative functional words.
    
    Args:
        tokens: List of word tokens.
        custom_stopwords: Optional override set of stop words.
    """
    sw = custom_stopwords if custom_stopwords is not None else STOPWORDS_SET
    return [t for t in tokens if t.lower() not in sw and len(t) > 1]

def apply_stemming(tokens: List[str]) -> List[str]:
    """Reduce tokens to their crude root base using Porter's rule-based stemmer.
    
    Note: Stemming can produce non-dictionary words (e.g. 'univers' for 'university')
    due to heuristic suffix stripping.
    """
    return [stemmer.stem(t) for t in tokens]

def apply_lemmatization(tokens: List[str]) -> List[str]:
    """Morphologically analyze tokens to find their true vocabulary lemma.
    
    Uses WordNet lemmatizer when available, falling back to stemmer or original token.
    """
    if lemmatizer:
        try:
            return [lemmatizer.lemmatize(t) for t in tokens]
        except Exception:
            return [stemmer.stem(t) for t in tokens]
    return [stemmer.stem(t) for t in tokens]

def preprocess_for_tfidf(text: str) -> str:
    """Pipeline tailored for traditional Bag-of-Words and TF-IDF representations.
    
    Steps:
    1. Normalization (lowercase, strip excessive symbols)
    2. Tokenization
    3. Stopword removal (reduces sparse matrix noise)
    4. Lemmatization (collapses inflected forms to dictionary headword)
    5. Re-join into a space-separated string suitable for TfidfVectorizer
    """
    normalized = normalize_text(text, lowercase=True, remove_punctuation=True)
    tokens = tokenize_text(normalized)
    filtered = remove_stopwords(tokens)
    lemmatized = apply_lemmatization(filtered)
    return " ".join(lemmatized)

def preprocess_for_embeddings(text: str) -> str:
    """Pipeline tailored for Transformer-based Dense Semantic Embeddings.
    
    Transformers rely on self-attention across syntax, sentence structure,
    and capitalization. Punctuation and stop words provide vital contextual
    cues, so we preserve them while only cleaning irregular whitespace.
    """
    if not text:
        return ""
    # Normalize unicode and normalize whitespace
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def explain_preprocessing_pipeline(text: str) -> Dict[str, Any]:
    """Educational demonstration of every stage in text transformation.
    
    Returns a dictionary tracing the step-by-step evolution of input text.
    """
    norm = normalize_text(text, lowercase=True, remove_punctuation=False)
    tokens = tokenize_text(norm)
    no_stops = remove_stopwords(tokens)
    stemmed = apply_stemming(no_stops)
    lemmatized = apply_lemmatization(no_stops)
    
    return {
        "original_text": text,
        "normalized_text": norm,
        "tokens": tokens,
        "stopwords_removed": no_stops,
        "stemmed_tokens": stemmed,
        "lemmatized_tokens": lemmatized,
        "tfidf_ready_string": " ".join(lemmatized),
        "embeddings_ready_string": preprocess_for_embeddings(text),
        "statistics": {
            "original_char_count": len(text),
            "original_word_count": len(text.split()),
            "token_count": len(tokens),
            "filtered_token_count": len(no_stops),
            "reduction_percentage": round((1.0 - (len(no_stops) / max(len(tokens), 1))) * 100, 2)
        }
    }
