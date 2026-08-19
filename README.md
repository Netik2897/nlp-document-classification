# Document Classification NLP Prototype (Python)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning and Natural Language Processing (NLP) document classification system built in Python. Designed for practical document and data processing use cases relevant to intelligent process automation (IPA) — including automatic routing and indexing of business documents such as **Invoices**, **Legal Contracts**, **Resumes/CVs**, **Customer Support Tickets**, and **Technical Specifications**.

---

## 🌟 Key Features

- **Text Cleaning & Preprocessing**: Lowercasing, noise filtering (URLs, emails, special characters), stop-word removal, and customizable regex token cleaning.
- **Feature Extraction**: Flexible TF-IDF vectorization (`TfidfVectorizer`) and Bag-of-Words (`CountVectorizer`) with custom n-gram ranges `(1, 2)`.
- **Model Classifier Pipeline**: Built-in integration with multiple algorithms:
  - Multinomial Naive Bayes (`MultinomialNB`)
  - Logistic Regression (`LogisticRegression`)
  - Random Forest Classifier (`RandomForestClassifier`)
  - Linear Support Vector Machine (`LinearSVC`)
- **Comprehensive Evaluation**: Metrics calculation for **Accuracy**, **Macro/Weighted Precision**, **Recall**, **F1-Score**, **Confusion Matrix**, and per-class performance reports.
- **Production Persistence**: Model serialization and deserialization via `joblib` pipelines.
- **CLI & Demo Interface**: Simple command-line interface to train, evaluate, benchmark, and predict classifications on raw text.
- **Unit Testing**: Automated unit tests using `pytest` / `unittest` covering preprocessing, feature extraction, model fitting, and metrics.

---

## 📁 Repository Structure

```text
nlp/
├── data/                       # Dataset export/import folder
├── models/                     # Serialized trained model pipelines (.joblib)
├── src/                        # Source modules
│   ├── __init__.py             # Package initializer
│   ├── dataset.py              # Synthetic document dataset generator
│   ├── preprocessing.py        # Text cleaning routines & sklearn transformer
│   ├── feature_extraction.py   # TF-IDF & Bag-of-Words vectorizer factory
│   ├── model.py                # Model training, sklearn pipeline, & persistence
│   └── evaluation.py           # Standard metrics & classification reports
├── tests/                      # Automated test suite
│   └── test_pipeline.py        # Unit tests for NLP pipeline
├── main.py                     # Command line interface & demo workflow
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quick Start

### 1. Setup Environment

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

### 2. Run Interactive Demo

Execute the demo workflow to benchmark candidate algorithms, evaluate performance, and classify sample documents:

```bash
python main.py --mode demo
```

**Sample Output:**

```text
======================================================================
      DOCUMENT CLASSIFICATION NLP PROTOTYPE - DEMO WORKFLOW
======================================================================

1. Generating synthetic document dataset for intelligent automation...
   Generated 250 total document samples across 5 categories.

2. Dataset split: 187 training samples, 63 test samples.

3. Model Benchmark & Comparison across NLP Classifiers...
   - naive_bayes          | Test Accuracy: 100.00% | Macro F1-Score: 1.0000
   - logistic_regression  | Test Accuracy: 100.00% | Macro F1-Score: 1.0000
   - random_forest        | Test Accuracy:  98.41% | Macro F1-Score: 0.9839
   - linear_svc           | Test Accuracy: 100.00% | Macro F1-Score: 1.0000

4. Best performing model selected: 'logistic_regression' (100.00% Accuracy)

5. Detailed Evaluation Metrics Report for Best Model:
============================================================
       EVALUATION METRICS SUMMARY - LOGISTIC_REGRESSION
============================================================
 Accuracy           : 1.0000 (100.00%)
 Precision (Macro)  : 1.0000
 Recall (Macro)     : 1.0000
 F1-Score (Macro)   : 1.0000
...
```

---

### 3. Custom Training

Train a specific model architecture and save it to disk:

```bash
python main.py --mode train --model logistic_regression --vectorizer tfidf --model-path models/best_document_classifier.joblib
```

---

### 4. Custom Inference / Classification

Classify any custom document text string using the trained pipeline:

```bash
python main.py --mode predict --text "INVOICE #94821. Total Amount: $4,500. Payment terms 30 days. Remit to ACME Corp."
```

**Output:**

```text
Input Document Text:
"INVOICE #94821. Total Amount: $4,500. Payment terms 30 days. Remit to ACME Corp."

Predicted Category: Invoice
```

---

## 🧪 Running Unit Tests

Run automated tests to verify preprocessing, vectorization, training, and metric calculations:

```bash
python -m unittest discover tests/
```

Or using `pytest`:

```bash
pytest tests/
```

---

## 💼 Resume / Project Highlights

- **Domain**: Machine Learning / Natural Language Processing (NLP) / Intelligent Automation
- **Stack**: Python 3.10+, Scikit-Learn, Pandas, NumPy, Joblib, Pytest
- **Impact**: Prototype demonstrates end-to-end NLP pipeline automation capable of categorizing unstructured business text into structured workflows with high accuracy and low inference latency.
