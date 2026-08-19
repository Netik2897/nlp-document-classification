"""
Flask REST API and Web Dashboard for NLP Document Classification.
Exposes JSON endpoints for intelligent document automation and an interactive UI.
"""

import os
import time
from flask import Flask, request, jsonify, render_template_string
from src.model import load_model, create_pipeline
from src.dataset import generate_sample_documents, CATEGORIES

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "best_document_classifier.joblib")
pipeline = None

def get_or_load_pipeline():
    global pipeline
    if pipeline is not None:
        return pipeline
        
    if os.path.exists(MODEL_PATH):
        pipeline = load_model(MODEL_PATH)
    else:
        # Fallback quick train if model file does not exist yet
        df = generate_sample_documents(samples_per_category=30, random_seed=42)
        pipeline = create_pipeline(model_name="logistic_regression")
        pipeline.fit(df['text'], df['category'])
        os.makedirs(os.path.dirname(MODEL_PATH) or ".", exist_ok=True)
        from src.model import save_model
        save_model(pipeline, MODEL_PATH)
        
    return pipeline

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intelligent Document Classification Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; font-size: 24px; }
        p.subtitle { color: #7f8c8d; font-size: 14px; margin-bottom: 25px; }
        textarea { width: 100%; height: 140px; padding: 12px; border: 1px solid #cccccc; border-radius: 6px; font-size: 14px; font-family: inherit; box-sizing: border-box; }
        .btn-group { margin-top: 15px; display: flex; gap: 10px; }
        button { background-color: #3498db; color: white; border: none; padding: 12px 24px; border-radius: 6px; font-size: 15px; font-weight: bold; cursor: pointer; transition: background 0.2s; }
        button:hover { background-color: #2980b9; }
        button.secondary { background-color: #95a5a6; }
        button.secondary:hover { background-color: #7f8c8d; }
        .result-box { margin-top: 25px; padding: 20px; border-radius: 8px; background: #eef7fc; border-left: 5px solid #3498db; display: none; }
        .result-title { font-size: 18px; font-weight: bold; color: #2c3e50; }
        .badge { display: inline-block; padding: 6px 12px; background: #2ecc71; color: white; border-radius: 20px; font-size: 14px; font-weight: bold; margin-left: 10px; }
        .meta-info { margin-top: 10px; font-size: 13px; color: #555; }
        .sample-buttons { margin-bottom: 15px; display: flex; flex-wrap: wrap; gap: 8px; }
        .sample-btn { background: #ecf0f1; color: #34495e; border: 1px solid #bdc3c7; padding: 6px 12px; border-radius: 4px; font-size: 12px; cursor: pointer; }
        .sample-btn:hover { background: #d5dbdb; }
    </style>
</head>
<body>
<div class="container">
    <h1>📄 Intelligent Document Classification Prototype</h1>
    <p class="subtitle">NLP Text Preprocessing • TF-IDF Feature Extraction • Machine Learning Classification</p>
    
    <label><strong>Try Quick Samples:</strong></label>
    <div class="sample-buttons">
        <button class="sample-btn" onclick="setSample('invoice')">Invoice Sample</button>
        <button class="sample-btn" onclick="setSample('contract')">Legal Contract</button>
        <button class="sample-btn" onclick="setSample('resume')">Resume / CV</button>
        <button class="sample-btn" onclick="setSample('ticket')">Support Ticket</button>
        <button class="sample-btn" onclick="setSample('spec')">Tech Spec</button>
    </div>

    <label for="documentText"><strong>Enter Unstructured Document Text:</strong></label>
    <textarea id="documentText" placeholder="Paste unstructured business document text here..."></textarea>
    
    <div class="btn-group">
        <button onclick="classifyDocument()">Classify Document</button>
        <button class="secondary" onclick="clearText()">Clear</button>
    </div>

    <div id="resultBox" class="result-box">
        <div class="result-title">
            Predicted Document Category: <span id="predictedCategory" class="badge">---</span>
        </div>
        <div class="meta-info">
            <strong>Confidence Level:</strong> <span id="confidenceScore">N/A</span> | 
            <strong>Processing Latency:</strong> <span id="latencyMs">0 ms</span>
        </div>
    </div>
</div>

<script>
const SAMPLES = {
    invoice: "INVOICE #94821 Date: 2026-03-01. Due Date: 2026-03-31. Total Amount Due: $4,500.00. Payment terms 30 days. Bill To: Acme Corp. Services: Cloud Consulting $4000, Tax $500.",
    contract: "MUTUAL NON-DISCLOSURE AGREEMENT. This Agreement is entered into by and between Acme Corp and Party B. Confidential Information shall mean all non-public technical and financial info.",
    resume: "Curriculum Vitae: Senior Data Scientist & ML Engineer. 5+ years experience in Python, PyTorch, Scikit-Learn, NLP, document classification, BERT, and AWS. B.S. Computer Science.",
    ticket: "Support Ticket #4029: Unable to reset account password. When clicking Forgot Password, no email link is received. User email: user@example.com. Priority: High.",
    spec: "TECHNICAL SYSTEM SPECIFICATION: REST API Endpoint Architecture. POST /api/v1/documents/classify. Request payload JSON containing raw text. Auth via Bearer Token."
};

function setSample(type) {
    document.getElementById('documentText').value = SAMPLES[type] || '';
}

function clearText() {
    document.getElementById('documentText').value = '';
    document.getElementById('resultBox').style.display = 'none';
}

async function classifyDocument() {
    const text = document.getElementById('documentText').value.trim();
    if (!text) {
        alert('Please enter document text to classify.');
        return;
    }
    
    try {
        const response = await fetch('/api/v1/classify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();
        
        if (data.status === 'success') {
            document.getElementById('predictedCategory').innerText = data.category;
            document.getElementById('confidenceScore').innerText = data.confidence !== null ? (data.confidence * 100).toFixed(1) + '%' : 'N/A';
            document.getElementById('latencyMs').innerText = data.latency_ms + ' ms';
            document.getElementById('resultBox').style.display = 'block';
        } else {
            alert('Error: ' + data.message);
        }
    } catch (err) {
        alert('Failed to connect to classification API: ' + err.message);
    }
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """Renders interactive Web UI Dashboard."""
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/v1/classify", methods=["POST"])
def classify():
    """
    JSON API endpoint for document classification.
    
    Payload:
        { "text": "Raw document text string..." }
        
    Response:
        {
            "status": "success",
            "category": "Invoice",
            "confidence": 0.85,
            "latency_ms": 12.4
        }
    """
    start_time = time.time()
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    
    if not text:
        return jsonify({"status": "error", "message": "Field 'text' is required in JSON payload."}), 400
        
    p = get_or_load_pipeline()
    category = p.predict([text])[0]
    
    confidence = None
    if hasattr(p.named_steps['classifier'], "predict_proba"):
        probas = p.predict_proba([text])[0]
        confidence = float(max(probas))
        
    latency_ms = round((time.time() - start_time) * 1000, 2)
    
    return jsonify({
        "status": "success",
        "category": category,
        "confidence": confidence,
        "latency_ms": latency_ms
    })

@app.route("/api/v1/categories", methods=["GET"])
def categories():
    """Returns list of supported document categories."""
    return jsonify({"categories": CATEGORIES})

if __name__ == "__main__":
    get_or_load_pipeline()
    print("Starting Document Classification REST API and Dashboard on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
    
