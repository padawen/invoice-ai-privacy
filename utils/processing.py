import os
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from .ocr import OCRProcessor
from .llm import OllamaClient
from .progress import progress_tracker
from config import Config

logger = logging.getLogger(__name__)

class InvoiceProcessor:
    """Main processing pipeline that combines OCR and LLM"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.ocr_processor = OCRProcessor(config)
        self.llm_client = OllamaClient(config)

    def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        try:
            llm_healthy = self.llm_client.health_check()

            return {
                "status": "healthy" if llm_healthy else "unhealthy",
                "components": {
                    "ocr": "healthy",
                    "llm": "healthy" if llm_healthy else "unhealthy",
                    "model": self.config.OLLAMA_MODEL
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

    def process_pdf(self, pdf_bytes: bytes, filename: str = "invoice.pdf", job_id: str = None) -> Dict[str, Any]:
        """
        Process PDF invoice through OCR + LLM pipeline

        Args:
            pdf_bytes: PDF file as bytes
            filename: Original filename (for logging)
            job_id: Progress tracking job ID

        Returns:
            Structured invoice data dictionary matching OpenAI format
        """
        processing_start = datetime.utcnow()
        logger.info(f"Starting processing of {filename}")

        try:
            # Update progress: Setup complete (5%)
            if job_id:
                progress_tracker.update_progress(job_id, "upload", 5, "File uploaded, starting OCR extraction", "processing")

                # Check for cancellation
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during setup")
                    return self.get_cancellation_result(filename, job_id)

            # Step 1: Extract text using OCR (5% - 20%)
            logger.info("Step 1: OCR text extraction...")
            if job_id:
                progress_tracker.update_progress(job_id, "ocr", 8, "Starting OCR processing", "processing")

            ocr_start = datetime.utcnow()
            extracted_text = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id)
            ocr_duration = (datetime.utcnow() - ocr_start).total_seconds()

            if not extracted_text or len(extracted_text.strip()) < 10:
                if job_id:
                    progress_tracker.set_error(job_id, "OCR extraction failed or returned insufficient text")
                raise Exception("OCR extraction failed or returned insufficient text")

            logger.info(f"OCR completed in {ocr_duration:.2f}s, extracted {len(extracted_text)} characters")
            if job_id:
                progress_tracker.update_progress(job_id, "ocr", 20, f"OCR completed, extracted {len(extracted_text)} characters", "processing")
                progress_tracker.update_stage_duration(job_id, "ocr", ocr_duration)

                # Check for cancellation after OCR
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled after OCR")
                    return self.get_cancellation_result(filename, job_id)

            # Step 2: Process text with LLM (20% - 90%)
            logger.info("Step 2: LLM structure extraction (chunked strategy)...")
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 25, "Starting LLM processing", "processing")

            llm_start = datetime.utcnow()

            # Always use chunking strategy for better speed and accuracy
            logger.info(f"Extracting invoice data in 2 chunks ({len(extracted_text)} chars)")

            # Chunk 1: Extract metadata
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 30, "Extracting metadata", "processing")

            metadata_prompt = self.llm_client.create_extraction_prompt(extracted_text, "metadata")
            metadata = self.llm_client.generate_completion(metadata_prompt, job_id)

            # Chunk 2: Extract line items
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 60, "Extracting line items", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during chunked processing")
                    return self.get_cancellation_result(filename, job_id)

            items_prompt = self.llm_client.create_extraction_prompt(extracted_text, "items")
            items_data = self.llm_client.generate_completion(items_prompt, job_id, expect_array=True)

            # Combine results
            if isinstance(items_data, list):
                metadata["invoice_data"] = items_data
            elif isinstance(items_data, dict) and "invoice_data" in items_data:
                metadata["invoice_data"] = items_data["invoice_data"]
            else:
                metadata["invoice_data"] = []

            structured_data = metadata
            llm_duration = (datetime.utcnow() - llm_start).total_seconds()

            logger.info(f"LLM processing completed in {llm_duration:.2f}s")
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 90, "LLM processing completed", "processing")
                progress_tracker.update_stage_duration(job_id, "llm", llm_duration)

            # Step 3: Post-process and validate (90% - 100%)
            logger.info("Step 3: Post-processing...")
            if job_id:
                progress_tracker.update_progress(job_id, "postprocess", 95, "Post-processing results", "processing")

            final_data = self.post_process_result(structured_data)

            # Add processing metadata
            total_duration = (datetime.utcnow() - processing_start).total_seconds()
            final_data["_processing_metadata"] = {
                "filename": filename,
                "method": "privacy_pipeline",
                "job_id": job_id,
                "ocr_duration": ocr_duration,
                "llm_duration": llm_duration,
                "total_duration": total_duration,
                "model": self.config.OLLAMA_MODEL,
                "processed_at": processing_start.isoformat()
            }

            logger.info(f"Processing completed successfully in {total_duration:.2f}s")

            # Complete progress tracking
            if job_id:
                progress_tracker.update_progress(job_id, "postprocess", 100, "Processing completed successfully", "completed")
                progress_tracker.update_stage_duration(job_id, "postprocess", (datetime.utcnow() - processing_start).total_seconds() - ocr_duration - llm_duration)
                progress_tracker.set_result(job_id, final_data)

            return final_data

        except Exception as e:
            error_msg = f"Processing failed for {filename}: {str(e)}"
            logger.error(error_msg)

            # Update progress with error
            if job_id:
                progress_tracker.set_error(job_id, error_msg)

            # Return fallback structure with error info
            return self.get_error_fallback(error_msg, filename)

    def post_process_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process the result to match OpenAI format exactly

        Args:
            data: Raw structured data from LLM

        Returns:
            Post-processed data matching expected format
        """
        try:
            # Add unique ID if not present
            if "id" not in data:
                data["id"] = str(uuid.uuid4())

            # Format dates consistently
            for date_field in ["issue_date", "due_date", "fulfillment_date"]:
                if date_field in data and data[date_field]:
                    data[date_field] = self.format_date_for_input(data[date_field])

            # Ensure all required fields are present
            data = self.ensure_required_structure(data)

            # Clean and validate numeric values in invoice_data
            if "invoice_data" in data and isinstance(data["invoice_data"], list):
                data["invoice_data"] = [
                    self.clean_invoice_item(item)
                    for item in data["invoice_data"]
                    if isinstance(item, dict)
                ]

            return data

        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            return self.get_error_fallback(f"Post-processing error: {str(e)}")

    def ensure_required_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields are present with correct structure"""

        # Default structure matching OpenAI format
        default_structure = {
            "id": str(uuid.uuid4()),
            "seller": {
                "name": "",
                "address": "",
                "tax_id": "",
                "email": "",
                "phone": ""
            },
            "buyer": {
                "name": "",
                "address": "",
                "tax_id": ""
            },
            "invoice_number": "",
            "issue_date": "",
            "fulfillment_date": "",
            "due_date": "",
            "payment_method": "",
            "currency": "",
            "invoice_data": []
        }

        # Merge with actual data, preserving structure
        for key, default_value in default_structure.items():
            if key not in data:
                data[key] = default_value
            elif key in ["seller", "buyer"] and isinstance(default_value, dict):
                # Ensure nested objects have all required fields
                if not isinstance(data[key], dict):
                    data[key] = default_value
                else:
                    for subkey, subdefault in default_value.items():
                        if subkey not in data[key]:
                            data[key][subkey] = subdefault

        return data

    def clean_invoice_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate a single invoice item"""
        cleaned_item = {
            "name": str(item.get("name", "")),
            "quantity": self.clean_numeric_string(item.get("quantity", "")),
            "unit_price": self.clean_numeric_string(item.get("unit_price", "")),
            "net": self.clean_numeric_string(item.get("net", "")),
            "gross": self.clean_numeric_string(item.get("gross", "")),
            "currency": str(item.get("currency", ""))
        }

        # Normalize currency
        if cleaned_item["currency"]:
            cleaned_item["currency"] = self.normalize_currency(cleaned_item["currency"])

        return cleaned_item

    def clean_numeric_string(self, value: Any) -> str:
        """Clean numeric values to match expected format"""
        if not value:
            return ""

        try:
            # Convert to string if not already
            str_value = str(value).strip()

            # Remove currency symbols and spaces
            cleaned = str_value.replace("Ft", "").replace("HUF", "").replace("EUR", "").replace("USD", "")
            cleaned = cleaned.replace(" ", "").replace("\u00a0", "")  # Regular and non-breaking spaces

            # Convert comma decimal separator to period
            if "," in cleaned and "." not in cleaned:
                cleaned = cleaned.replace(",", ".")

            # Remove any remaining non-numeric characters except periods and minus
            import re
            cleaned = re.sub(r'[^\d.-]', '', cleaned)

            return cleaned

        except Exception as e:
            logger.warning(f"Failed to clean numeric value '{value}': {str(e)}")
            return ""

    def normalize_currency(self, currency: str) -> str:
        """Normalize currency codes"""
        currency_lower = currency.lower().strip()

        if currency_lower in ["ft", "huf", "forint"]:
            return "HUF"
        elif currency_lower in ["eur", "€", "euro"]:
            return "EUR"
        elif currency_lower in ["usd", "$", "dollar"]:
            return "USD"
        elif currency_lower in ["gbp", "£", "pound"]:
            return "GBP"

        return currency.upper()

    def format_date_for_input(self, date_string: str) -> str:
        """Format date string for input fields (YYYY-MM-DD)"""
        if not date_string:
            return ""

        try:
            # Common date formats to try
            date_formats = [
                "%Y-%m-%d",      # 2024-01-15
                "%Y.%m.%d",      # 2024.01.15
                "%Y. %m. %d.",   # 2025. 01. 10.
                "%Y. %m. %d",    # 2025. 01. 10
                "%Y / %m / %d",  # 2025 / 01 / 10
                "%Y/%m/%d",      # 2024/01/15
                "%d.%m.%Y",      # 15.01.2024
                "%d. %m. %Y.",   # 15. 01. 2024.
                "%d. %m. %Y",    # 15. 01. 2024
                "%d/%m/%Y",      # 15/01/2024
                "%d-%m-%Y",      # 15-01-2024
                "%m/%d/%Y",      # 01/15/2024
                "%Y%m%d",        # 20240115
            ]

            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_string.strip(), fmt)
                    return parsed_date.strftime("%Y-%m-%d")
                except ValueError:
                    continue

            # If no format matches, return original
            logger.warning(f"Could not parse date: {date_string}")
            return date_string

        except Exception as e:
            logger.warning(f"Date formatting error: {str(e)}")
            return date_string

    def get_error_fallback(self, error_message: str, filename: str = "unknown") -> Dict[str, Any]:
        """Return fallback structure when processing fails"""
        return {
            "id": str(uuid.uuid4()),
            "error": error_message,
            "seller": {
                "name": "",
                "address": "",
                "tax_id": "",
                "email": "",
                "phone": ""
            },
            "buyer": {
                "name": "",
                "address": "",
                "tax_id": ""
            },
            "invoice_number": "",
            "issue_date": "",
            "fulfillment_date": "",
            "due_date": "",
            "payment_method": "",
            "currency": "HUF",
            "invoice_data": [],
            "_processing_metadata": {
                "filename": filename,
                "method": "privacy_pipeline",
                "error": error_message,
                "processed_at": datetime.utcnow().isoformat()
            }
        }

    def get_cancellation_result(self, filename: str, job_id: str = None) -> Dict[str, Any]:
        """Return structure when processing is cancelled"""
        return {
            "id": str(uuid.uuid4()),
            "cancelled": True,
            "message": "Processing was cancelled by user",
            "seller": {
                "name": "",
                "address": "",
                "tax_id": "",
                "email": "",
                "phone": ""
            },
            "buyer": {
                "name": "",
                "address": "",
                "tax_id": ""
            },
            "invoice_number": "",
            "issue_date": "",
            "fulfillment_date": "",
            "due_date": "",
            "payment_method": "",
            "currency": "HUF",
            "invoice_data": [],
            "_processing_metadata": {
                "filename": filename,
                "method": "privacy_pipeline",
                "job_id": job_id,
                "cancelled": True,
                "processed_at": datetime.utcnow().isoformat()
            }
        }