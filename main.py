"""
Main entry point for Document Classification NLP Prototype.
Provides CLI for training models, model evaluation, and predicting document categories.
"""

import argparse
import sys
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset import generate_sample_documents, CATEGORIES
from src.model import create_pipeline, save_model, load_model, SUPPORTED_MODELS
from src.evaluation import evaluate_model, print_evaluation_summary
from src.preprocessing import clean_text

def run_demo():
    """
    Executes a complete demonstration of the document classification prototype.
    1. Generates synthetic multi-class document dataset.
    2. Compares multiple NLP classification algorithms.
    3. Trains the best model pipeline.
    4. Evaluates performance using standard classification metrics.
    5. Demonstrates intelligent document routing predictions on sample inputs.
    """
    print("\n" + "="*70)
    print("      DOCUMENT CLASSIFICATION NLP PROTOTYPE - DEMO WORKFLOW")
    print("="*70)
    
    print("\n1. Generating synthetic document dataset for intelligent automation...")
    df = generate_sample_documents(samples_per_category=50, random_seed=42)
    print(f"   Generated {len(df)} total document samples across {len(CATEGORIES)} categories:")
    for cat, count in df['category'].value_counts().items():
        print(f"    - {cat:<25}: {count} documents")
        
    # Split into Train / Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.25, random_state=42, stratify=df['category']
    )
    print(f"\n2. Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")
    
    print("\n3. Model Benchmark & Comparison across NLP Classifiers...")
    results = {}
    best_model_name = None
    best_acc = 0.0
    best_pipeline = None
    
    for model_name in SUPPORTED_MODELS.keys():
        pipeline = create_pipeline(model_name=model_name, vectorizer_method="tfidf", ngram_range=(1, 2))
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, labels=CATEGORIES)
        acc = metrics['accuracy']
        f1 = metrics['f1_macro']
        results[model_name] = (acc, f1)
        print(f"   - {model_name:<20} | Test Accuracy: {acc*100:6.2f}% | Macro F1-Score: {f1:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_name
            best_pipeline = pipeline

    print(f"\n4. Best performing model selected: '{best_model_name}' ({best_acc*100:.2f}% Accuracy)")
    
    print("\n5. Detailed Evaluation Metrics Report for Best Model:")
    y_pred_best = best_pipeline.predict(X_test)
    best_metrics = evaluate_model(y_test, y_pred_best, labels=CATEGORIES)
    print_evaluation_summary(best_metrics, model_name=best_model_name)
    
    # Save the trained model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "best_document_classifier.joblib")
    save_model(best_pipeline, model_path)
    print(f"\n6. Model pipeline saved to disk: '{model_path}'")
    
    print("\n7. Intelligent Document Processing (Inference Demonstration):")
    sample_documents = [
        "INVOICE #94821. Billing Amount $4,250.00 USD. Payment terms 30 days. Remit to ACME Financial Corp.",
        "Candidate Resume: Experienced Software Engineer specializing in Python, NLP algorithms, scikit-learn, and microservice deployment.",
        "System Spec: RESTful API definition for document ingestion webhook. Method POST, Content-Type application/json.",
        "Support Ticket: App keeps crashing on submit with error 500. User cannot log in to dashboard. Please investigate.",
        "MUTUAL NON-DISCLOSURE AGREEMENT. Confidentiality terms, governing law of Delaware, non-solicitation clause for 2 years."
    ]
    
    print("-" * 70)
    for i, doc in enumerate(sample_documents, 1):
        pred_label = best_pipeline.predict([doc])[0]
        # Predict class probabilities if model supports it
        proba_str = ""
        if hasattr(best_pipeline.named_steps['classifier'], "predict_proba"):
            probas = best_pipeline.predict_proba([doc])[0]
            confidence = max(probas) * 100
            proba_str = f" (Confidence: {confidence:.1f}%)"
            
        print(f" Document #{i}: \"{doc[:65]}...\"")
        print(f" ==> Classified Category: [{pred_label}]{proba_str}\n")
    print("=" * 70)
    print("Demo completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Python NLP Document Classification Prototype")
    parser.add_argument("--mode", choices=["demo", "train", "predict"], default="demo",
                        help="Execution mode: demo (full pipeline run), train (train model), predict (classify input text)")
    parser.add_argument("--model", choices=list(SUPPORTED_MODELS.keys()), default="logistic_regression",
                        help="Classifier model architecture")
    parser.add_argument("--vectorizer", choices=["tfidf", "bow"], default="tfidf",
                        help="Feature vectorization technique")
    parser.add_argument("--text", type=str, help="Text string to classify in 'predict' mode")
    parser.add_argument("--model-path", type=str, default="models/best_document_classifier.joblib",
                        help="Path to saved model pipeline file")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "train":
        print(f"Training '{args.model}' classifier using '{args.vectorizer}' vectorizer...")
        df = generate_sample_documents(samples_per_category=60, random_seed=42)
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
        )
        pipeline = create_pipeline(model_name=args.model, vectorizer_method=args.vectorizer)
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, labels=CATEGORIES)
        print_evaluation_summary(metrics, model_name=args.model)
        
        os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
        save_model(pipeline, args.model_path)
        print(f"Model saved to '{args.model_path}'")
        
    elif args.mode == "predict":
        if not args.text:
            print("Error: --text parameter required for 'predict' mode.", file=sys.stderr)
            sys.exit(1)
            
        if os.path.exists(args.model_path):
            pipeline = load_model(args.model_path)
            print(f"Loaded trained pipeline from '{args.model_path}'")
        else:
            print(f"Warning: Model file '{args.model_path}' not found. Training quick fallback model...")
            df = generate_sample_documents(samples_per_category=30)
            pipeline = create_pipeline(model_name="logistic_regression")
            pipeline.fit(df['text'], df['category'])
            
        prediction = pipeline.predict([args.text])[0]
        print("\nInput Document Text:")
        print(f"\"{args.text}\"")
        print(f"\nPredicted Category: {prediction}")

if __name__ == "__main__":
    main()
