import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
from config import config
from utils.processing import InvoiceProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app(config_name=None):
    """Application factory"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Configure CORS
    CORS(app, origins="*")

    # Initialize configuration
    config[config_name].init_app(app)

    # Initialize processor
    processor = InvoiceProcessor(config[config_name])

    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(e):
        return jsonify({
            'error': 'File too large. Maximum size is 50MB.'
        }), 413

    @app.errorhandler(Exception)
    def handle_general_error(e):
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'fallbackData': get_fallback_data()
        }), 500

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        try:
            health_status = processor.health_check()
            status_code = 200 if health_status.get('status') == 'healthy' else 503
            return jsonify(health_status), status_code
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e)
            }), 503

    @app.route('/process-invoice', methods=['POST'])
    def process_invoice():
        """
        Process invoice PDF - compatible with OpenAI endpoints
        Accepts same input format as /api/processTextPDF and /api/processImagePDF
        """
        try:
            # Validate authentication if API key is configured
            if app.config.get('API_KEY'):
                auth_header = request.headers.get('Authorization', '')
                api_key = auth_header.replace('Bearer ', '').replace('bearer ', '')

                if not api_key or api_key != app.config['API_KEY']:
                    return jsonify({'error': 'Unauthorized'}), 401

            # Get file from request
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            # Validate file type
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({'error': 'Only PDF files are allowed'}), 400

            # Read file content
            pdf_bytes = file.read()
            if len(pdf_bytes) == 0:
                return jsonify({'error': 'Uploaded file is empty'}), 400

            # Check file size
            if len(pdf_bytes) > app.config['MAX_FILE_SIZE']:
                return jsonify({'error': 'File too large (max 50MB)'}), 400

            logger.info(f"Processing file: {file.filename} ({len(pdf_bytes)} bytes)")

            # Process the PDF
            result = processor.process_pdf(pdf_bytes, file.filename)

            # Handle processing errors
            if 'error' in result:
                logger.warning(f"Processing completed with error: {result['error']}")
                return jsonify(result), 500

            # Remove processing metadata for external response (keep it in logs)
            if '_processing_metadata' in result:
                metadata = result.pop('_processing_metadata')
                logger.info(f"Processing metadata: {metadata}")

            logger.info(f"Successfully processed {file.filename}")
            return jsonify(result), 200

        except RequestEntityTooLarge:
            return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            return jsonify({
                'error': str(e),
                'fallbackData': get_fallback_data()
            }), 500

    @app.route('/', methods=['GET'])
    def index():
        """Root endpoint with service information"""
        return jsonify({
            'service': 'Invoice AI Privacy',
            'version': '1.0.0',
            'description': 'Privacy-focused invoice processing using OCR and local LLM',
            'endpoints': {
                'health': '/health',
                'process': '/process-invoice'
            },
            'status': 'operational'
        })

    def get_fallback_data():
        """Return fallback data structure matching OpenAI format"""
        import uuid
        return {
            'id': str(uuid.uuid4()),
            'seller': {
                'name': '',
                'address': '',
                'tax_id': '',
                'email': '',
                'phone': ''
            },
            'buyer': {
                'name': '',
                'address': '',
                'tax_id': ''
            },
            'invoice_number': '',
            'issue_date': '',
            'fulfillment_date': '',
            'due_date': '',
            'payment_method': '',
            'currency': 'HUF',
            'invoice_data': []
        }

    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    app.run(
        host=app.config.get('HOST', '0.0.0.0'),
        port=app.config.get('PORT', 5000),
        debug=app.config.get('DEBUG', False)
    )