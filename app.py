import os
import logging
import threading
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils.processing import InvoiceProcessor
from utils.progress import progress_tracker
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

# Initialize config and processor
Config.init_app(app)
processor = InvoiceProcessor(Config)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        health_data = processor.health_check()
        return jsonify(health_data)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route('/process-invoice', methods=['POST'])
def process_invoice():
    """Process invoice PDF and extract data using Ollama"""
    try:
        # Check API key if configured
        if app.config.get('API_KEY'):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Missing or invalid authorization header"}), 401

            token = auth_header.split(' ')[1]
            if token != app.config['API_KEY']:
                return jsonify({"error": "Invalid API key"}), 401

        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": "Only PDF files are supported"}), 400

        # Check file size
        file.seek(0, 2)  # Seek to end of file
        file_size = file.tell()
        file.seek(0)  # Seek back to beginning

        if file_size > app.config['MAX_FILE_SIZE']:
            return jsonify({
                "error": f"File too large. Maximum size: {app.config['MAX_FILE_SIZE']} bytes"
            }), 400

        # Get the file bytes
        filename = secure_filename(file.filename)
        pdf_bytes = file.read()

        logger.info(f"Processing invoice: {filename} ({file_size} bytes)")

        # Create progress tracking job
        job_id = progress_tracker.create_job(filename)
        logger.info(f"Created job {job_id} for {filename}")

        # Start background processing
        def background_process():
            try:
                processor.process_pdf(pdf_bytes, filename, job_id)
                logger.info(f"Successfully processed invoice: {filename}")
            except Exception as e:
                logger.error(f"Background processing failed: {str(e)}")
                progress_tracker.set_error(job_id, str(e))

        # Start processing in background thread
        thread = threading.Thread(target=background_process)
        thread.daemon = True
        thread.start()

        # Return job ID immediately for progress tracking
        return jsonify({
            "_processing_metadata": {
                "job_id": job_id,
                "filename": filename,
                "status": "processing",
                "message": "Processing started, use job_id to track progress"
            }
        })

    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/progress/<job_id>', methods=['GET'])
def get_progress(job_id):
    """Get progress for a specific job"""
    try:
        # Check API key if configured
        if app.config.get('API_KEY'):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Missing or invalid authorization header"}), 401

            token = auth_header.split(' ')[1]
            if token != app.config['API_KEY']:
                return jsonify({"error": "Invalid API key"}), 401

        progress = progress_tracker.get_progress(job_id)
        if progress is None:
            return jsonify({"error": "Job not found"}), 404

        return jsonify(progress), 200

    except Exception as e:
        logger.error(f"Error getting progress for job {job_id}: {str(e)}")
        return jsonify({
            "error": "Failed to get progress",
            "details": str(e)
        }), 500

@app.route('/cancel-job/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a processing job"""
    try:
        # Check API key if configured
        if app.config.get('API_KEY'):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({"error": "Missing or invalid authorization header"}), 401

            token = auth_header.split(' ')[1]
            if token != app.config['API_KEY']:
                return jsonify({"error": "Invalid API key"}), 401

        success = progress_tracker.cancel_job(job_id)
        if not success:
            return jsonify({"error": "Job not found or cannot be cancelled"}), 404

        logger.info(f"Job {job_id} cancelled successfully")
        return jsonify({"message": "Job cancelled successfully"}), 200

    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {str(e)}")
        return jsonify({
            "error": "Failed to cancel job",
            "details": str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint with service info"""
    return jsonify({
        "service": "Invoice AI Privacy",
        "version": "1.0.0",
        "model": app.config['OLLAMA_MODEL'],
        "endpoints": {
            "health": "/health",
            "process": "/process-invoice",
            "progress": "/progress/<job_id>"
        },
        "features": [
            "PDF invoice processing",
            "OCR text extraction",
            "AI-powered data extraction",
            "Secure API key authentication"
        ]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info(f"Starting Invoice AI Privacy service...")
    logger.info(f"Model: {app.config['OLLAMA_MODEL']}")
    logger.info(f"Ollama Host: {app.config['OLLAMA_HOST']}")
    logger.info(f"API Key: {'Configured' if app.config.get('API_KEY') else 'Not configured'}")
    logger.info(f"Temp Directory: {app.config['TEMP_DIR']}")

    app.run(
        host=app.config['HOST'],
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )