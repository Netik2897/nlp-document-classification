"""
Feature extraction module for document classification.
Configures Bag-of-Words (CountVectorizer) and TF-IDF (TfidfVectorizer) representations.
"""

from typing import Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vectorizer(
    method: str = "tfidf",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: Optional[int] = 5000,
    min_df: int = 1
):
    """
    Factory function to instantiate text feature vectorizers.
    
    Args:
        method: Vectorization method ('tfidf' or 'bow'/'count').
        ngram_range: (min_n, max_n) n-gram tuple.
        max_features: Maximum number of vocabulary features.
        min_df: Minimum document frequency threshold.
        
    Returns:
        CountVectorizer or TfidfVectorizer instance.
    """
    method = method.lower()
    if method in ["tfidf", "tf-idf"]:
        return TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True
        )
    elif method in ["bow", "count", "bag_of_words"]:
        return CountVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df
        )
    else:
        raise ValueError(f"Unsupported vectorization method: '{method}'. Choose 'tfidf' or 'bow'.")
