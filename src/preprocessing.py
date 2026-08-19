"""
Text cleaning and preprocessing module for NLP document classification.
Implements custom cleaning routines and scikit-learn compatible transformers.
"""

import re
from typing import List, Union
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Standard English stop words set for standalone text cleaning
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should",
    "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}

def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_emails: bool = True,
    remove_numbers: bool = False,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True
) -> str:
    """
    Cleans raw document text by applying standard NLP cleaning steps.
    
    Args:
        text: Raw text string.
        lowercase: Convert text to lower case.
        remove_urls: Strip HTTP/HTTPS URLs.
        remove_emails: Strip email addresses.
        remove_numbers: Strip numerical digits.
        remove_punctuation: Strip punctuation and special characters.
        remove_stopwords: Remove standard English stop words.
        
    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
        
    if lowercase:
        text = text.lower()
        
    if remove_urls:
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
    if remove_emails:
        text = re.sub(r'\S+@\S+', ' ', text)
        
    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)
        
    if remove_punctuation:
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        tokens = text.split()
        tokens = [word for word in tokens if word not in DEFAULT_STOPWORDS]
        text = " ".join(tokens)
        
    return text

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for text cleaning within ML Pipelines.
    """
    def __init__(self, remove_stopwords: bool = True, remove_numbers: bool = False):
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: Union[List[str], pd.Series]) -> List[str]:
        return [
            clean_text(
                doc,
                remove_stopwords=self.remove_stopwords,
                remove_numbers=self.remove_numbers
            ) for doc in X
        ]
