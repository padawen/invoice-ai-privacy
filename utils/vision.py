import os
import base64
import logging
import requests
import json
import time
import threading
import warnings
from typing import Dict, Any
from PIL import Image
from pdf2image import convert_from_bytes
import io

# Suppress json_repair warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from json_repair import repair_json

from config import Config

logger = logging.getLogger(__name__)

class VisionProcessor:
    """Process invoices using vision LLM (llava)"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.base_url = f"http://{self.config.OLLAMA_HOST}"
        self.vision_model = self.config.VISION_MODEL
        self.timeout = self.config.OLLAMA_TIMEOUT
        logger.info(f"Vision processor initialized with model: {self.vision_model}")

    def extract_invoice_data(self, pdf_bytes: bytes, job_id: str = None) -> Dict[str, Any]:
        """
        Extract invoice data using vision model

        Args:
            pdf_bytes: PDF file as bytes
            job_id: Optional job ID for cancellation checking

        Returns:
            Dict containing structured invoice data
        """
        try:
            logger.info("Starting vision-based invoice extraction")

            # Check for cancellation
            if job_id:
                from .progress import progress_tracker
                if progress_tracker.is_cancelled(job_id):
                    raise Exception("Processing was cancelled by user")

            # Convert PDF to image (higher DPI for better OCR)
            images = convert_from_bytes(pdf_bytes, dpi=300, fmt='png')
            if not images:
                raise ValueError("Could not convert PDF to images")

            logger.info(f"Converted PDF to {len(images)} images")

            # Process first page (most invoices are single page)
            image = images[0]

            # Resize if too large (llava works better with smaller images)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {new_size}")

            # Convert to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG", optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            logger.info(f"Image size: {len(img_base64)} bytes (base64)")

            # Extract data using vision model
            extracted_data = self._vision_extract(img_base64, job_id)

            return extracted_data

        except Exception as e:
            logger.error(f"Vision extraction failed: {str(e)}")
            raise Exception(f"Vision extraction failed: {str(e)}")

    def _vision_extract(self, image_base64: str, job_id: str = None) -> Dict[str, Any]:
        """Extract invoice data from image using vision LLM"""

        prompt = """You are looking at an invoice image. Extract all the information you can see.

First, describe what you see in the invoice:
- Who is the seller (company name, address, tax ID)?
- Who is the buyer?
- What are the invoice numbers and dates?
- What items are being sold? List the table rows.

Then, for each item in the table, look at ALL the numbers in that row from left to right.
The last/rightmost number is usually the gross total (with tax).

Return your answer as JSON:
{"seller":{"name":"","address":"","tax_id":"","email":"","phone":""},"buyer":{"name":"","address":"","tax_id":""},"invoice_number":"","issue_date":"YYYY-MM-DD","fulfillment_date":"YYYY-MM-DD","due_date":"YYYY-MM-DD","payment_method":"","currency":"HUF","invoice_data":[{"name":"","quantity":"","unit_price":"","net":"","gross":"","currency":"HUF"}]}"""

        # Shared state for threading
        result = {'response': None, 'error': None, 'cancelled': False}
        response_obj = {'response': None}

        def vision_request():
            try:
                payload = {
                    "model": self.vision_model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": True,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024
                    }
                }

                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )

                response_obj['response'] = response

                if response.status_code != 200:
                    result['error'] = f"Vision API error: {response.status_code}"
                    return

                # Process streaming response
                raw_response = ""
                try:
                    for line in response.iter_lines(decode_unicode=True):
                        if result['cancelled']:
                            logger.info(f"Job {job_id} cancelled during vision processing")
                            break

                        if line.strip():
                            try:
                                chunk_data = json.loads(line)
                                if 'response' in chunk_data:
                                    raw_response += chunk_data['response']
                                if chunk_data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue

                    processing_time = time.time() - start_time
                    logger.info(f"Vision processing completed in {processing_time:.2f}s")
                    logger.info(f"Raw vision response ({len(raw_response)} chars):\n{raw_response[:500]}...")

                    if result['cancelled']:
                        result['error'] = "Processing was cancelled by user"
                    elif not raw_response:
                        result['error'] = "Empty response from vision model"
                    else:
                        # Parse JSON response
                        parsed_data = self._parse_response(raw_response)
                        result['response'] = parsed_data

                finally:
                    response.close()

            except Exception as e:
                result['error'] = str(e)

        # Start request in separate thread
        request_thread = threading.Thread(target=vision_request)
        request_thread.daemon = True
        request_thread.start()

        # Monitor for cancellation
        check_interval = 0.2
        while request_thread.is_alive():
            if job_id:
                from .progress import progress_tracker
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} cancellation requested")
                    result['cancelled'] = True

                    if response_obj['response']:
                        try:
                            response_obj['response'].close()
                        except:
                            pass

                    request_thread.join(timeout=3.0)
                    raise Exception("Processing was cancelled by user")

            time.sleep(check_interval)

        request_thread.join()

        if result['error']:
            raise Exception(result['error'])

        if result['response'] is None:
            raise Exception("No response from vision model")

        return result['response']

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse vision model response to extract JSON"""
        from json_repair import repair_json

        try:
            # Clean response
            cleaned = response_text.strip()

            # Remove markdown code blocks
            if cleaned.startswith('```json') and cleaned.endswith('```'):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith('```') and cleaned.endswith('```'):
                cleaned = cleaned[3:-3].strip()

            # Extract JSON object
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, cleaned, re.DOTALL)

            if matches:
                json_str = max(matches, key=len)
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try repair
                    repaired = repair_json(json_str)
                    parsed = json.loads(repaired)

                return parsed
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            logger.error(f"Failed to parse vision response: {str(e)}")
            logger.error(f"Raw response: {response_text}")

            # Return fallback structure
            return {
                "seller": {"name": "", "address": "", "tax_id": "", "email": "", "phone": ""},
                "buyer": {"name": "", "address": "", "tax_id": ""},
                "invoice_number": "",
                "issue_date": "",
                "fulfillment_date": "",
                "due_date": "",
                "payment_method": "",
                "currency": "",
                "invoice_data": []
            }
