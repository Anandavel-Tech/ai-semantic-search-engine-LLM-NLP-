"""Document Clustering and Unsupervised Analysis Module.
Performs K-Means clustering on TF-IDF or Dense Embeddings, generates 2D PCA projections,
identifies cluster keywords via centroid inspection, and calculates Silhouette Scores.
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer

from services.preprocessor import preprocess_for_tfidf

class ClusteringService:
    """Clustering engine supporting both sparse TF-IDF and dense embeddings."""
    
    def __init__(self, tfidf_service=None, embeddings_service=None):
        self.tfidf_service = tfidf_service
        self.embeddings_service = embeddings_service

    def cluster_documents(
        self,
        documents: List[Dict[str, Any]],
        num_clusters: int = 3,
        method: str = "tfidf"  # 'tfidf' or 'embeddings'
    ) -> Dict[str, Any]:
        """Perform K-Means clustering with 2D PCA projection and keyword extraction.
        
        Args:
            documents: List of document dicts with 'id', 'title', and 'full_text'.
            num_clusters: Requested number of clusters.
            method: Feature representation ('tfidf' or 'embeddings').
        """
        n_docs = len(documents)
        if n_docs < 2:
            return {
                "status": "error",
                "message": "At least 2 documents are required to perform clustering.",
                "clusters": []
            }
            
        k = max(2, min(num_clusters, n_docs - 1))
        texts = [d["full_text"] for d in documents]
        
        # 1. Extract feature representation
        if method.lower() == "embeddings" and self.embeddings_service:
            feature_matrix = self.embeddings_service.encode(texts)
            feature_names = None
        else:
            # TF-IDF representation
            vectorizer = TfidfVectorizer(
                max_features=2000, 
                ngram_range=(1, 2), 
                stop_words="english",
                sublinear_tf=True
            )
            clean_texts = [preprocess_for_tfidf(t) for t in texts]
            try:
                feature_matrix = vectorizer.fit_transform(clean_texts).toarray()
                feature_names = vectorizer.get_feature_names_out()
            except ValueError:
                return {
                    "status": "error",
                    "message": "Corpus does not contain sufficient unique terms for clustering.",
                    "clusters": []
                }
                
        # 2. Run K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_matrix)
        centroids = kmeans.cluster_centers_
        
        # 3. Calculate Silhouette Score
        try:
            sil_score = float(silhouette_score(feature_matrix, labels))
            sil_score = round(sil_score, 4)
        except Exception:
            sil_score = 0.0
            
        # 4. Compute 2D PCA projection for visualization
        n_components = min(2, feature_matrix.shape[1], feature_matrix.shape[0])
        pca = PCA(n_components=n_components, random_state=42)
        coords = pca.fit_transform(feature_matrix)
        
        # 5. Extract Cluster Keywords and Group Documents
        cluster_groups: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(k)}
        plot_points = []
        
        for idx, doc in enumerate(documents):
            cluster_id = int(labels[idx])
            point = {
                "document_id": doc["id"],
                "title": doc["title"],
                "cluster": cluster_id,
                "x": round(float(coords[idx][0]), 4),
                "y": round(float(coords[idx][1]) if coords.shape[1] > 1 else 0.0, 4),
                "word_count": doc.get("word_count", len(doc["full_text"].split()))
            }
            cluster_groups[cluster_id].append(doc)
            plot_points.append(point)
            
        # Determine Top Keywords per Cluster
        cluster_summaries = []
        for c_id in range(k):
            docs_in_c = cluster_groups[c_id]
            
            if feature_names is not None:
                # Centroid weights in TF-IDF space
                top_word_indices = np.argsort(centroids[c_id])[::-1][:6]
                keywords = [feature_names[i] for i in top_word_indices]
            else:
                # For dense embeddings, collect most frequent non-stopwords in cluster
                combined_c_text = " ".join(d["full_text"] for d in docs_in_c)
                v = TfidfVectorizer(max_features=6, stop_words="english")
                try:
                    v.fit([combined_c_text])
                    keywords = list(v.get_feature_names_out())
                except Exception:
                    keywords = ["topic", "cluster", "content"]
                    
            cluster_summaries.append({
                "cluster_id": c_id,
                "label": f"Cluster {c_id + 1}: {', '.join(keywords[:3])}",
                "size": len(docs_in_c),
                "percentage": round((len(docs_in_c) / n_docs) * 100, 1),
                "keywords": keywords,
                "documents": [
                    {"id": d["id"], "title": d["title"]}
                    for d in docs_in_c
                ]
            })
            
        return {
            "status": "success",
            "method": method.upper(),
            "num_clusters": k,
            "total_documents": n_docs,
            "silhouette_score": sil_score,
            "explained_variance_ratio": [round(float(v), 3) for v in pca.explained_variance_ratio_],
            "clusters": cluster_summaries,
            "plot_points": plot_points
        }
