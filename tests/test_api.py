"""
Unit test suite for Flask REST API endpoints.
"""

import unittest
import json
from app import app, get_or_load_pipeline

class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        get_or_load_pipeline()

    def test_home_dashboard(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Intelligent Document Classification Dashboard", response.data)

    def test_categories_endpoint(self):
        response = self.app.get('/api/v1/categories')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("categories", data)
        self.assertIn("Invoice", data["categories"])

    def test_classify_endpoint_success(self):
        payload = {
            "text": "INVOICE #9812 Total Amount $500 Payment terms 30 days due by 2026-04-01"
        }
        response = self.app.post(
            '/api/v1/classify',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["category"], "Invoice")
        self.assertIn("latency_ms", data)

    def test_classify_endpoint_missing_text(self):
        payload = {}
        response = self.app.post(
            '/api/v1/classify',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "error")

if __name__ == "__main__":
    unittest.main()
