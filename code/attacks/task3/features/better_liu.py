"""Better Liu/Semantic detector — extended sentence-embedding features.

Aktualne branch_d ma 4 features (adj_cosine, LSH KL-div). Tutaj bardziej
bogaty zestaw, wszystkie z all-MiniLM-L6-v2 (już cached).

Liu's Semantic Invariant Robust Watermark embeds tokens których embedding
cluster jest constrained by previous sentence's embedding. Detection idea:
  - sentence embeddings układają się w dziwny pattern dla watermarked text
  - velocity (delta between adjacent embeddings) jest restricted
  - clustering structure ujawnia non-random groupings
"""
from __future__ import annotations

import re

import numpy as np

_sbert = None


def _load_sbert():
    global _sbert
    if _sbert is None:
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer("all-MiniLM-L6-v2")
    return _sbert


def _split_sentences(text: str, min_words: int = 4) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p.split()) >= min_words]


def extract(text: str) -> dict[str, float]:
    feats: dict[str, float] = {}

    sents = _split_sentences(text)
    feats["liu_n_sents"] = float(len(sents))

    if len(sents) < 3:
        zeros = ["liu_adj_cos_mean", "liu_adj_cos_std", "liu_adj_cos_min", "liu_adj_cos_max",
                 "liu_velocity_mean", "liu_velocity_std",
                 "liu_global_cos_mean", "liu_global_cos_std",
                 "liu_kmeans_inertia", "liu_kmeans_silhouette",
                 "liu_anomaly_max", "liu_anomaly_mean",
                 "liu_embed_norm_mean", "liu_embed_norm_std"]
        for k in zeros:
            feats[k] = 0.0
        return feats

    model = _load_sbert()
    embeds = model.encode(sents, normalize_embeddings=True, show_progress_bar=False)

    # ── Adjacent cosine similarity
    adj_cos = np.array([float(np.dot(embeds[i], embeds[i + 1])) for i in range(len(embeds) - 1)])
    feats["liu_adj_cos_mean"] = float(adj_cos.mean())
    feats["liu_adj_cos_std"] = float(adj_cos.std())
    feats["liu_adj_cos_min"] = float(adj_cos.min())
    feats["liu_adj_cos_max"] = float(adj_cos.max())

    # ── Velocity (norm of delta vectors)
    velocities = np.linalg.norm(np.diff(embeds, axis=0), axis=1)
    feats["liu_velocity_mean"] = float(velocities.mean())
    feats["liu_velocity_std"] = float(velocities.std())

    # ── Global cosine (each sentence vs document mean)
    doc_mean = embeds.mean(axis=0)
    doc_mean /= np.linalg.norm(doc_mean) + 1e-9
    global_cos = embeds @ doc_mean
    feats["liu_global_cos_mean"] = float(global_cos.mean())
    feats["liu_global_cos_std"] = float(global_cos.std())

    # ── Embedding norm stats (already normalized but check decoded)
    norms = np.linalg.norm(embeds, axis=1)
    feats["liu_embed_norm_mean"] = float(norms.mean())
    feats["liu_embed_norm_std"] = float(norms.std())

    # ── K-means clustering (K=3 to match potential 3 watermark types or topic clusters)
    if len(sents) >= 4:
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            k = min(3, len(sents) - 1)
            km = KMeans(n_clusters=k, n_init=3, random_state=42)
            labels = km.fit_predict(embeds)
            feats["liu_kmeans_inertia"] = float(km.inertia_) / len(sents)  # normalized
            if len(set(labels)) > 1 and len(sents) >= 3:
                feats["liu_kmeans_silhouette"] = float(silhouette_score(embeds, labels))
            else:
                feats["liu_kmeans_silhouette"] = 0.0
        except Exception:
            feats["liu_kmeans_inertia"] = 0.0
            feats["liu_kmeans_silhouette"] = 0.0
    else:
        feats["liu_kmeans_inertia"] = 0.0
        feats["liu_kmeans_silhouette"] = 0.0

    # ── Anomaly: distance of each sentence from text centroid
    centroid_dists = np.linalg.norm(embeds - doc_mean / np.linalg.norm(doc_mean + 1e-9), axis=1)
    feats["liu_anomaly_max"] = float(centroid_dists.max())
    feats["liu_anomaly_mean"] = float(centroid_dists.mean())

    return feats
