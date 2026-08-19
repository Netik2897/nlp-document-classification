"""
Model training, serialization, and inference pipeline module.
Supports Multinomial Naive Bayes, Logistic Regression, Random Forest, and Linear SVC classifiers.
"""

import joblib
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from .preprocessing import TextCleaner
from .feature_extraction import get_vectorizer

SUPPORTED_MODELS = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "linear_svc": LinearSVC(random_state=42)
}

def create_pipeline(
    model_name: str = "logistic_regression",
    vectorizer_method: str = "tfidf",
    ngram_range: tuple = (1, 2),
    max_features: Optional[int] = 5000
) -> Pipeline:
    """
    Creates an end-to-end scikit-learn NLP Classification Pipeline.
    
    Args:
        model_name: Name of classifier model.
        vectorizer_method: 'tfidf' or 'bow'.
        ngram_range: n-gram range tuple.
        max_features: Max vocabulary size.
        
    Returns:
        Pipeline: Sklearn Pipeline object.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model name: '{model_name}'. Choose from {list(SUPPORTED_MODELS.keys())}")
        
    cleaner = TextCleaner(remove_stopwords=True)
    vectorizer = get_vectorizer(method=vectorizer_method, ngram_range=ngram_range, max_features=max_features)
    classifier = SUPPORTED_MODELS[model_name]
    
    pipeline = Pipeline([
        ('cleaner', cleaner),
        ('vectorizer', vectorizer),
        ('classifier', classifier)
    ])
    
    return pipeline

def save_model(pipeline: Pipeline, filepath: str) -> None:
    """Saves trained pipeline object to disk using joblib."""
    joblib.dump(pipeline, filepath)

def load_model(filepath: str) -> Pipeline:
    """Loads trained pipeline object from disk using joblib."""
    return joblib.load(filepath)
