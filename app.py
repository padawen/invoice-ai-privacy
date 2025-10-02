import os
import logging
import threading
import time
from flask import Flask, request, jsonify, Response, stream_with_context
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

# Suppress werkzeug logging for progress endpoints to reduce noise
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Only log errors from Flask, not every request

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
            # Support both x-api-key and Authorization: Bearer headers
            api_key = request.headers.get('x-api-key')
            auth_header = request.headers.get('Authorization')

            if api_key:
                if api_key != app.config['API_KEY']:
                    return jsonify({"error": "Invalid API key"}), 401
            elif auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                if token != app.config['API_KEY']:
                    return jsonify({"error": "Invalid API key"}), 401
            else:
                return jsonify({"error": "Missing or invalid authorization header"}), 401

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
    """Get progress for a specific job (polling - use SSE instead to reduce requests)"""
    try:
        # Check API key if configured
        if app.config.get('API_KEY'):
            # Support both x-api-key and Authorization: Bearer headers
            api_key = request.headers.get('x-api-key')
            auth_header = request.headers.get('Authorization')

            if api_key:
                # x-api-key header
                if api_key != app.config['API_KEY']:
                    return jsonify({"error": "Invalid API key"}), 401
            elif auth_header and auth_header.startswith('Bearer '):
                # Authorization: Bearer header
                token = auth_header.split(' ')[1]
                if token != app.config['API_KEY']:
                    return jsonify({"error": "Invalid API key"}), 401
            else:
                return jsonify({"error": "Missing or invalid authorization header"}), 401

        progress = progress_tracker.get_progress(job_id)
        if progress is None:
            return jsonify({"error": "Job not found"}), 404

        # Don't log these requests - they're frequent polling
        return jsonify(progress), 200

    except Exception as e:
        logger.error(f"Error getting progress for job {job_id}: {str(e)}")
        return jsonify({
            "error": "Failed to get progress",
            "details": str(e)
        }), 500

@app.route('/progress-stream/<job_id>', methods=['GET'])
def progress_stream(job_id):
    """Server-Sent Events stream for real-time progress updates (recommended over polling)"""
    def generate():
        import json

        # Check API key - support both x-api-key and Authorization: Bearer headers
        if app.config.get('API_KEY'):
            api_key = request.headers.get('x-api-key')
            auth_header = request.headers.get('Authorization')

            valid_auth = False
            if api_key and api_key == app.config['API_KEY']:
                valid_auth = True
            elif auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                if token == app.config['API_KEY']:
                    valid_auth = True

            if not valid_auth:
                yield f"event: error\ndata: {json.dumps({'error': 'Invalid authorization'})}\n\n"
                return

        # Stream progress updates
        last_progress = None
        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            progress = progress_tracker.get_progress(job_id)

            if progress is None:
                yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            # Only send updates when progress changes
            if progress != last_progress:
                yield f"event: progress\ndata: {json.dumps(progress)}\n\n"
                last_progress = progress

            # Check if done
            if progress.get('status') in ['completed', 'failed']:
                yield f"event: complete\ndata: {json.dumps(progress)}\n\n"
                break

            time.sleep(0.5)  # Check every 500ms

        # Timeout
        if time.time() - start_time >= max_wait:
            yield f"event: timeout\ndata: {json.dumps({'error': 'Stream timeout'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/cancel-job/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a processing job"""
    try:
        # Check API key if configured
        if app.config.get('API_KEY'):
            # Support both x-api-key and Authorization: Bearer headers
            api_key = request.headers.get('x-api-key')
            auth_header = request.headers.get('Authorization')

            if api_key:
                if api_key != app.config['API_KEY']:
                    return jsonify({"error": "Invalid API key"}), 401
            elif auth_header and auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
                if token != app.config['API_KEY']:
                    return jsonify({"error": "Invalid API key"}), 401
            else:
                return jsonify({"error": "Missing or invalid authorization header"}), 401

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