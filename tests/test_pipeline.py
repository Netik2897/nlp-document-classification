"""
Unit test suite for Document Classification NLP Prototype.
"""

import os
import unittest
import pandas as pd
from src.dataset import generate_sample_documents, CATEGORIES
from src.preprocessing import clean_text, TextCleaner
from src.feature_extraction import get_vectorizer
from src.model import create_pipeline, save_model, load_model
from src.evaluation import evaluate_model

class TestNLPPipeline(unittest.TestCase):

    def test_clean_text(self):
        raw_text = "INVOICE #94821 Date: 2026-03-01. Total Amount: $450.00! Please visit https://acme.com/pay"
        cleaned = clean_text(raw_text, remove_stopwords=True)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("#", cleaned)
        self.assertIn("invoice", cleaned.split()) # 'invoice' is retained after lowercasing
        self.assertTrue(cleaned.islower())

    def test_text_cleaner_transformer(self):
        cleaner = TextCleaner()
        docs = ["Invoice #123!", "Payment Due Date: 2026."]
        transformed = cleaner.transform(docs)
        self.assertEqual(len(transformed), 2)
        self.assertIsInstance(transformed[0], str)

    def test_dataset_generation(self):
        df = generate_sample_documents(samples_per_category=10, random_seed=123)
        self.assertEqual(len(df), 50)
        self.assertIn('text', df.columns)
        self.assertIn('category', df.columns)
        unique_cats = set(df['category'])
        self.assertEqual(unique_cats, set(CATEGORIES))

    def test_feature_extraction(self):
        vectorizer = get_vectorizer(method="tfidf", ngram_range=(1, 2))
        corpus = ["invoice payment balance due", "resume skills python ML", "contract agreement terms"]
        X = vectorizer.fit_transform(corpus)
        self.assertEqual(X.shape[0], 3)
        self.assertGreater(X.shape[1], 0)

    def test_pipeline_fit_predict(self):
        df = generate_sample_documents(samples_per_category=5, random_seed=42)
        pipeline = create_pipeline(model_name="naive_bayes", vectorizer_method="tfidf")
        pipeline.fit(df['text'], df['category'])
        
        test_doc = "INVOICE #888 Total $500 Payment Due"
        pred = pipeline.predict([test_doc])
        self.assertEqual(len(pred), 1)
        self.assertIn(pred[0], CATEGORIES)

    def test_model_serialization(self):
        df = generate_sample_documents(samples_per_category=5, random_seed=42)
        pipeline = create_pipeline(model_name="logistic_regression")
        pipeline.fit(df['text'], df['category'])
        
        test_path = "test_model_temp.joblib"
        try:
            save_model(pipeline, test_path)
            self.assertTrue(os.path.exists(test_path))
            
            loaded_pipeline = load_model(test_path)
            pred_orig = pipeline.predict(["Agreement terms contract"])[0]
            pred_loaded = loaded_pipeline.predict(["Agreement terms contract"])[0]
            self.assertEqual(pred_orig, pred_loaded)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

    def test_evaluation_metrics(self):
        y_true = ["Invoice", "Invoice", "Legal Contract", "Resume / CV"]
        y_pred = ["Invoice", "Legal Contract", "Legal Contract", "Resume / CV"]
        metrics = evaluate_model(y_true, y_pred)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("precision_macro", metrics)
        self.assertIn("recall_macro", metrics)
        self.assertIn("f1_macro", metrics)
        self.assertEqual(metrics["accuracy"], 0.75)

if __name__ == "__main__":
    unittest.main()
