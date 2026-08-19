"""
Model evaluation module.
Calculates standard classification metrics and generates formatted performance reports.
"""

from typing import Dict, Any
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, labels=None) -> Dict[str, Any]:
    """
    Computes standard document classification evaluation metrics.
    
    Args:
        y_true: True category labels.
        y_pred: Predicted category labels.
        labels: List of category label strings.
        
    Returns:
        dict: Metric results dictionary.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Macro metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted metrics
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    report_str = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in labels], columns=[f"Pred: {l}" for l in labels])
    
    metrics = {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1_macro": f1_macro,
        "precision_weighted": p_weighted,
        "recall_weighted": r_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report_str,
        "confusion_matrix": cm_df
    }
    
    return metrics

def print_evaluation_summary(metrics: Dict[str, Any], model_name: str = "Model") -> None:
    """Prints a clean, formatted evaluation report to stdout."""
    print("=" * 60)
    print(f"       EVALUATION METRICS SUMMARY - {model_name.upper()}")
    print("=" * 60)
    print(f" Accuracy           : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f" Precision (Macro)  : {metrics['precision_macro']:.4f}")
    print(f" Recall (Macro)     : {metrics['recall_macro']:.4f}")
    print(f" F1-Score (Macro)   : {metrics['f1_macro']:.4f}")
    print(f" Precision (Weight) : {metrics['precision_weighted']:.4f}")
    print(f" Recall (Weight)    : {metrics['recall_weighted']:.4f}")
    print(f" F1-Score (Weight)  : {metrics['f1_weighted']:.4f}")
    print("-" * 60)
    print("Detailed Classification Report:")
    print(metrics['classification_report'])
    print("-" * 60)
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("=" * 60)
