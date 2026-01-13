"""
Semantic word clustering utilities for wordcloud generation.

This module provides functionality to cluster words semantically and
arrange them in the wordcloud based on semantic similarity.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict

from .logging_config import get_logger

logger = get_logger(__name__)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.debug("scikit-learn not available, semantic clustering disabled")

try:
    import gensim
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.debug("gensim not available, Word2Vec clustering disabled")


def cluster_words_tfidf(
    words: List[str],
    frequencies: List[float],
    num_clusters: Optional[int] = None,
    method: str = 'kmeans'
) -> Dict[str, int]:
    """
    Cluster words using TF-IDF vectorization and clustering algorithms.
    
    Args:
        words: List of words to cluster
        frequencies: List of frequencies corresponding to words
        num_clusters: Number of clusters (auto-determined if None)
        method: Clustering method ('kmeans' or 'hierarchical')
        
    Returns:
        Dictionary mapping words to cluster IDs
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for semantic clustering")
    
    if not words:
        return {}
    
    # Auto-determine number of clusters
    if num_clusters is None:
        num_clusters = min(10, max(2, len(words) // 5))
    
    num_clusters = min(num_clusters, len(words))
    
    # Create TF-IDF vectors
    # Use words as documents (with repetition based on frequency for better clustering)
    documents = []
    for word, freq in zip(words, frequencies):
        # Repeat word based on frequency to weight it
        repeat_count = max(1, int(freq * 10))
        documents.extend([word] * repeat_count)
    
    try:
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=100)
        vectors = vectorizer.fit_transform(documents)
        
        # Cluster
        if method == 'hierarchical':
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            cluster_labels = clustering.fit_predict(vectors.toarray())
        else:  # kmeans
            clustering = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            cluster_labels = clustering.fit_predict(vectors.toarray())
        
        # Map words to clusters (use most common cluster for each word)
        word_clusters = defaultdict(list)
        word_idx = 0
        for word, freq in zip(words, frequencies):
            repeat_count = max(1, int(freq * 10))
            word_labels = cluster_labels[word_idx:word_idx + repeat_count]
            # Use most common cluster
            from collections import Counter
            most_common_cluster = Counter(word_labels).most_common(1)[0][0]
            word_clusters[word] = most_common_cluster
            word_idx += repeat_count
        
        logger.debug(f"Clustered {len(words)} words into {num_clusters} clusters")
        return dict(word_clusters)
        
    except Exception as e:
        logger.error(f"TF-IDF clustering failed: {e}")
        # Fallback: assign all words to cluster 0
        return {word: 0 for word in words}


def cluster_words_similarity(
    words: List[str],
    similarity_threshold: float = 0.3
) -> Dict[str, int]:
    """
    Cluster words based on string similarity (Levenshtein distance).
    
    Args:
        words: List of words to cluster
        similarity_threshold: Minimum similarity to be in same cluster
        
    Returns:
        Dictionary mapping words to cluster IDs
    """
    if not words:
        return {}
    
    def levenshtein_distance(s1: str, s2: str) -> float:
        """Calculate normalized Levenshtein distance."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        max_len = max(len(s1), len(s2))
        return previous_row[-1] / max_len if max_len > 0 else 0.0
    
    def similarity(s1: str, s2: str) -> float:
        """Calculate similarity (1 - normalized distance)."""
        return 1.0 - levenshtein_distance(s1, s2)
    
    # Cluster words
    clusters: Dict[str, int] = {}
    cluster_id = 0
    unassigned = set(words)
    
    while unassigned:
        # Start new cluster with first unassigned word
        seed_word = unassigned.pop()
        clusters[seed_word] = cluster_id
        
        # Find similar words
        similar_words = [seed_word]
        remaining = list(unassigned)
        
        for word in remaining:
            # Check similarity with any word in current cluster
            max_sim = max(similarity(word, sw) for sw in similar_words)
            if max_sim >= similarity_threshold:
                clusters[word] = cluster_id
                similar_words.append(word)
                unassigned.remove(word)
        
        cluster_id += 1
    
    logger.debug(f"Clustered {len(words)} words into {cluster_id} clusters using similarity")
    return clusters


def arrange_words_by_cluster(
    words: List[Tuple[str, float, int]],
    clusters: Dict[str, int],
    cluster_layout: str = 'grouped'
) -> List[Tuple[str, float, int]]:
    """
    Rearrange words based on cluster assignments.
    
    Args:
        words: List of (word, normalized_freq, original_freq) tuples
        clusters: Dictionary mapping words to cluster IDs
        cluster_layout: Layout strategy ('grouped', 'interleaved', 'sorted')
        
    Returns:
        Rearranged list of word tuples
    """
    if not clusters:
        return words
    
    # Group words by cluster
    cluster_groups: Dict[int, List[Tuple[str, float, int]]] = defaultdict(list)
    unclustered: List[Tuple[str, float, int]] = []
    
    for word, freq, count in words:
        cluster_id = clusters.get(word)
        if cluster_id is not None:
            cluster_groups[cluster_id].append((word, freq, count))
        else:
            unclustered.append((word, freq, count))
    
    # Sort clusters by total frequency
    cluster_order = sorted(
        cluster_groups.keys(),
        key=lambda cid: sum(freq for _, freq, _ in cluster_groups[cid]),
        reverse=True
    )
    
    # Arrange based on layout strategy
    result = []
    
    if cluster_layout == 'grouped':
        # Group all words from same cluster together
        for cluster_id in cluster_order:
            # Sort within cluster by frequency
            cluster_words = sorted(cluster_groups[cluster_id], key=lambda x: x[1], reverse=True)
            result.extend(cluster_words)
        result.extend(unclustered)
    
    elif cluster_layout == 'interleaved':
        # Interleave words from different clusters
        max_cluster_size = max(len(cluster_groups[cid]) for cid in cluster_order) if cluster_order else 0
        
        for i in range(max_cluster_size):
            for cluster_id in cluster_order:
                if i < len(cluster_groups[cluster_id]):
                    result.append(cluster_groups[cluster_id][i])
        result.extend(unclustered)
    
    else:  # 'sorted' - sort by cluster then frequency
        for cluster_id in cluster_order:
            cluster_words = sorted(cluster_groups[cluster_id], key=lambda x: x[1], reverse=True)
            result.extend(cluster_words)
        result.extend(sorted(unclustered, key=lambda x: x[1], reverse=True))
    
    return result


def get_cluster_colors(
    clusters: Dict[str, int],
    color_theme: str = 'viridis'
) -> Dict[str, str]:
    """
    Assign colors to words based on their cluster membership.
    
    Args:
        clusters: Dictionary mapping words to cluster IDs
        color_theme: Color theme name
        
    Returns:
        Dictionary mapping words to color strings
    """
    from .visualization import COLOR_THEMES, generate_colors_by_frequency
    
    if not clusters:
        return {}
    
    # Get unique cluster IDs
    unique_clusters = sorted(set(clusters.values()))
    num_clusters = len(unique_clusters)
    
    # Get colors from theme
    theme_colors = COLOR_THEMES.get(color_theme, COLOR_THEMES['viridis'])
    
    # Assign colors to clusters
    cluster_colors: Dict[int, str] = {}
    for i, cluster_id in enumerate(unique_clusters):
        color_idx = (i * len(theme_colors)) // num_clusters
        cluster_colors[cluster_id] = theme_colors[min(color_idx, len(theme_colors) - 1)]
    
    # Map words to colors
    word_colors = {word: cluster_colors[cluster_id] for word, cluster_id in clusters.items()}
    
    return word_colors

