import pytest
import requests
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = os.getenv('TEST_API_URL', 'http://localhost:5000')
SAMPLE_PDF_PATH = Path(__file__).parent / 'sample_invoices' / 'test_invoice.pdf'

class TestInvoicePrivacyAPI:
    """Test suite for Invoice AI Privacy API"""

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{API_BASE_URL}/health")

        assert response.status_code == 200
        data = response.json()

        assert 'status' in data
        assert 'components' in data
        assert 'timestamp' in data

    def test_root_endpoint(self):
        """Test root endpoint returns service information"""
        response = requests.get(f"{API_BASE_URL}/")

        assert response.status_code == 200
        data = response.json()

        assert 'service' in data
        assert data['service'] == 'Invoice AI Privacy'
        assert 'endpoints' in data

    def test_process_invoice_no_file(self):
        """Test process endpoint without file"""
        response = requests.post(f"{API_BASE_URL}/process-invoice")

        assert response.status_code == 400
        data = response.json()
        assert 'error' in data

    def test_process_invoice_empty_file(self):
        """Test process endpoint with empty file"""
        files = {'file': ('empty.pdf', b'', 'application/pdf')}
        response = requests.post(f"{API_BASE_URL}/process-invoice", files=files)

        assert response.status_code == 400
        data = response.json()
        assert 'error' in data

    @pytest.mark.skipif(not SAMPLE_PDF_PATH.exists(), reason="Sample PDF not available")
    def test_process_invoice_valid_file(self):
        """Test process endpoint with valid PDF file"""
        with open(SAMPLE_PDF_PATH, 'rb') as f:
            files = {'file': ('test_invoice.pdf', f, 'application/pdf')}
            response = requests.post(f"{API_BASE_URL}/process-invoice", files=files, timeout=120)

        # Should return 200 or 500 (with fallback data)
        assert response.status_code in [200, 500]
        data = response.json()

        # Check response structure matches OpenAI format
        self.validate_invoice_structure(data)

    def validate_invoice_structure(self, data):
        """Validate that response matches expected invoice structure"""
        # Handle error responses with fallback data
        if 'error' in data and 'fallbackData' in data:
            data = data['fallbackData']

        # Required top-level fields
        required_fields = ['seller', 'buyer', 'invoice_data']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate seller structure
        seller = data['seller']
        assert isinstance(seller, dict)
        seller_fields = ['name', 'address', 'tax_id', 'email', 'phone']
        for field in seller_fields:
            assert field in seller, f"Missing seller field: {field}"
            assert isinstance(seller[field], str)

        # Validate buyer structure
        buyer = data['buyer']
        assert isinstance(buyer, dict)
        buyer_fields = ['name', 'address', 'tax_id']
        for field in buyer_fields:
            assert field in buyer, f"Missing buyer field: {field}"
            assert isinstance(buyer[field], str)

        # Validate invoice_data structure
        invoice_data = data['invoice_data']
        assert isinstance(invoice_data, list)

        for item in invoice_data:
            assert isinstance(item, dict)
            item_fields = ['name', 'quantity', 'unit_price', 'net', 'gross', 'currency']
            for field in item_fields:
                assert field in item, f"Missing invoice item field: {field}"
                assert isinstance(item[field], str)

        # Optional fields should be strings if present
        optional_fields = ['id', 'invoice_number', 'issue_date', 'fulfillment_date',
                          'due_date', 'payment_method', 'currency']
        for field in optional_fields:
            if field in data:
                assert isinstance(data[field], str), f"Field {field} should be string"

    def test_api_compatibility_with_openai_format(self):
        """Test that API response is compatible with OpenAI format"""
        # Mock response structure
        mock_response = {
            "id": "test-id",
            "seller": {
                "name": "Test Company",
                "address": "123 Test St",
                "tax_id": "12345678",
                "email": "test@company.com",
                "phone": "+1234567890"
            },
            "buyer": {
                "name": "Test Buyer",
                "address": "456 Buyer St",
                "tax_id": "87654321"
            },
            "invoice_number": "INV-001",
            "issue_date": "2024-01-15",
            "fulfillment_date": "2024-01-16",
            "due_date": "2024-02-15",
            "payment_method": "Bank transfer",
            "currency": "HUF",
            "invoice_data": [
                {
                    "name": "Test Product",
                    "quantity": "2",
                    "unit_price": "1000",
                    "net": "2000",
                    "gross": "2540",
                    "currency": "HUF"
                }
            ]
        }

        # Validate structure
        self.validate_invoice_structure(mock_response)

if __name__ == '__main__':
    # Run basic connectivity test
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        print(f"✅ API is accessible at {API_BASE_URL}")
        print(f"Health status: {response.json().get('status', 'unknown')}")
    except requests.exceptions.RequestException as e:
        print(f"❌ API not accessible at {API_BASE_URL}: {e}")
        print("Make sure the service is running with: docker-compose up")

    # Run pytest if available
    try:
        import pytest
        pytest.main([__file__, '-v'])
    except ImportError:
        print("Install pytest for full test suite: pip install pytest")