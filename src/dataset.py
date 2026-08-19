"""
Dataset generator and manager for document classification.
Generates realistic domain-specific texts for intelligent document automation.
"""

import pandas as pd
import random
from typing import Tuple, List

# Categories relevant to intelligent automation use cases
CATEGORIES = [
    "Invoice",
    "Legal Contract",
    "Resume / CV",
    "Customer Support Ticket",
    "Technical Specification"
]

DATA_TEMPLATES = {
    "Invoice": [
        "INVOICE #{num} Date: {date} Due Date: {due_date}. Total Amount Due: ${amount}. Payment terms 30 days. Bill To: {company}. Items: Software License $2000, Maintenance $500. Tax: 10%. Thank you for your business.",
        "TAX INVOICE #{num}. Vendor: ACME Solutions Inc. Customer: {company}. PO Number: PO-89421. Outstanding Balance: ${amount}. Please remit payment via bank transfer. Late fees apply after 30 days.",
        "BILLING STATEMENT Invoice Reference: #{num}. Date of Issue: 2026-01-15. Amount: ${amount}. Subtotal $1200, Shipping $50, Total Payable ${amount}. Payment details enclosed.",
        "Purchase Order Payment Notice. Invoice #{num} total USD ${amount}. Services rendered: Cloud Migration & DevOps Consulting. Vendor Account: 9482-104-1. Due by {due_date}."
    ],
    "Legal Contract": [
        "MUTUAL NON-DISCLOSURE AGREEMENT. This Agreement is entered into by and between {company} and Party B. Confidential Information shall mean all non-public technical, financial, or business information disclosed. Governing law: State of Delaware.",
        "MASTER SERVICES AGREEMENT (MSA). Section 4: Limitation of Liability. Neither party shall be liable for indirect, incidental, or consequential damages. Effective Date: 2026-02-01. Termination requires 30 days written notice.",
        "TERMS OF SERVICE AND LICENSING AGREEMENT. Licensee is granted a non-exclusive, non-transferable license to use the software. Intellectual Property Rights remain solely with Licensor. Indemnification obligations set forth in Clause 8.",
        "EMPLOYMENT CONTRACT AGREEMENT. Employee agrees to non-solicitation and non-compete clauses for a period of 12 months following termination. Compensation and benefit packages detailed in Schedule A."
    ],
    "Resume / CV": [
        "Curriculum Vitae: Senior Data Scientist & ML Engineer. 5+ years experience in Python, PyTorch, Scikit-Learn, NLP, document classification, BERT, and AWS. Education: Master of Computer Science. Project experience in automated data extraction pipelines.",
        "RESUME - Professional Software Developer. Skills: Python, Java, SQL, REST APIs, Docker, Kubernetes, CI/CD. Developed microservices architecture and intelligent automation bots. B.S. in Software Engineering.",
        "Summary of Qualifications: Natural Language Processing Specialist. Experienced in text cleaning, TF-IDF vectorization, sentiment analysis, topic modeling, and spaCy. Key Achievements: Built document routing system achieving 95% accuracy.",
        "CANDIDATE PROFILE: Intelligent Automation Engineer. Expertise in Python scripting, Optical Character Recognition (OCR), document processing, regex parser development, and machine learning pipeline integration."
    ],
    "Customer Support Ticket": [
        "Ticket #4029: Unable to reset account password. When clicking 'Forgot Password', no email link is received. User email: user@example.com. Browser: Chrome. Priority: High.",
        "Support Inquiry: Billing error on monthly subscription. I was charged twice for order #98124. Please issue a refund for the duplicate charge of $49.99 as soon as possible.",
        "Issue Report: Application crashes on startup after updating to version 2.4. Error code ERR_CONNECTION_REFUSED. Please fix this bug immediately.",
        "Customer Help Ticket: Package delivery delay. Tracking number #849102 shows no updates for 4 days. Requesting status update or replacement shipment."
    ],
    "Technical Specification": [
        "TECHNICAL SYSTEM SPECIFICATION: REST API Endpoint Architecture. POST /api/v1/documents/classify. Request payload JSON containing raw text. Response return JSON schema: label, score, execution_time_ms. Auth via Bearer Token.",
        "SOFTWARE ARCHITECTURE DOCUMENT: Document Ingestion Pipeline. Ingestion module reads PDF files, triggers OCR engine, applies text preprocessing, converts to TF-IDF feature matrix, and passes to classifier model.",
        "SYSTEM REQUIREMENTS: Scalable Microservice Infrastructure. Minimum memory 4GB RAM, Python 3.10+, scikit-learn model loading latency < 50ms. Asynchronous queue handling via Celery and Redis.",
        "API INTEGRATION DOCUMENTATION: Authentication and Webhook Payloads. Endpoints require HMAC SHA256 signature verification. Error responses comply with RFC 7807 problem details specification."
    ]
}

def generate_sample_documents(samples_per_category: int = 40, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic multi-class document classification dataset.
    
    Args:
        samples_per_category: Number of document samples per category.
        random_seed: Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: DataFrame containing 'text' and 'category' columns.
    """
    random.seed(random_seed)
    companies = ["Acme Corp", "TechCorp Inc", "Global Logistics", "FinTech Solutions", "Apex Innovations", "Nexus Data"]
    dates = ["2026-03-01", "2026-03-15", "2026-04-01", "2026-05-10"]
    
    data = []
    for cat, templates in DATA_TEMPLATES.items():
        for i in range(samples_per_category):
            tmpl = random.choice(templates)
            doc_text = tmpl.format(
                num=random.randint(10000, 99999),
                date=random.choice(dates),
                due_date=random.choice(dates),
                amount=random.randint(150, 15000),
                company=random.choice(companies)
            )
            # Add minor random variations to simulate realistic text diversity
            variations = [
                " Reference ID: " + str(random.randint(100, 999)),
                " Priority level: " + random.choice(["Normal", "High", "Urgent"]),
                " Processed automatically by pipeline.",
                " Notes: Verified by automated classifier."
            ]
            doc_text += random.choice(variations)
            data.append({"text": doc_text, "category": cat})
            
    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df
