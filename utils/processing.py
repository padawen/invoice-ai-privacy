import os
import logging
import uuid
import re
import unicodedata
from typing import Dict, Any, Optional, List
from datetime import datetime
from .ocr import OCRProcessor
from .llm import OllamaClient
from .vision import VisionProcessor
from .progress import progress_tracker
from .price_mapper import price_mapper
from config import Config

logger = logging.getLogger(__name__)

class InvoiceProcessor:
    """Main processing pipeline that combines OCR and LLM"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.ocr_processor = OCRProcessor(config)
        self.llm_client = OllamaClient(config)
        self.vision_processor = VisionProcessor(config)

        # Use vision mode if enabled
        self.use_vision = self.config.USE_VISION_MODEL

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
        logger.info(f"Starting processing of {filename} (vision mode: {self.use_vision})")

        try:
            # Update progress: Upload complete
            upload_start = processing_start
            if job_id:
                progress_tracker.update_progress(job_id, "upload", 100, "File uploaded successfully", "processing")
                upload_duration = (datetime.utcnow() - upload_start).total_seconds()
                progress_tracker.update_stage_duration(job_id, "upload", upload_duration)

                # Check for cancellation
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during setup")
                    return self.get_cancellation_result(filename, job_id)

            # If vision mode enabled, use vision processor directly
            if self.use_vision:
                logger.info("Using vision-based extraction...")
                if job_id:
                    progress_tracker.update_progress(job_id, "vision", 10, "Starting vision processing", "processing")

                vision_start = datetime.utcnow()
                structured_data = self.vision_processor.extract_invoice_data(pdf_bytes, job_id)
                vision_duration = (datetime.utcnow() - vision_start).total_seconds()

                logger.info(f"Vision processing completed in {vision_duration:.2f}s")
                if job_id:
                    progress_tracker.update_progress(job_id, "vision", 90, "Vision processing completed", "processing")
                    progress_tracker.update_stage_duration(job_id, "vision", vision_duration)

                # Apply fixes
                if "invoice_data" in structured_data and isinstance(structured_data["invoice_data"], list):
                    structured_data["invoice_data"] = self.fix_net_gross_confusion(structured_data["invoice_data"])

                # Post-process and complete
                final_result = self.post_process_result(structured_data)
                processing_duration = (datetime.utcnow() - processing_start).total_seconds()

                if job_id:
                    progress_tracker.update_progress(job_id, "complete", 100, "Processing completed", "completed")
                    progress_tracker.set_result(job_id, final_result)

                logger.info(f"Processing completed successfully in {processing_duration:.2f}s")
                return final_result

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
            logger.info(f"Extracting invoice data in 2 chunks")

            # Chunk 1: Extract metadata (use structured OCR for better layout preservation - Priority 1 fix)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 30, "Extracting metadata with structured OCR", "processing")

            # Use structured OCR for metadata to preserve two-column layouts and field boundaries
            structured_text_metadata = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id, structured=True)
            logger.info(f"Using structured OCR for metadata extraction ({len(structured_text_metadata)} chars)")

            metadata_prompt = self.llm_client.create_extraction_prompt(structured_text_metadata, "metadata")
            metadata = self.llm_client.generate_completion(metadata_prompt, job_id)

            # Chunk 2: Extract line items (use structured OCR for table)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 60, "Extracting line items with structured OCR", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during chunked processing")
                    return self.get_cancellation_result(filename, job_id)

            # Re-OCR with table structure for items extraction
            structured_text = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id, structured=True)
            logger.info(f"Re-extracted structured text for items ({len(structured_text)} chars)")

            items_prompt = self.llm_client.create_extraction_prompt(structured_text, "items")
            # Changed to expect_array=False since items now returns {"invoice_data": [...]}
            items_data = self.llm_client.generate_completion(items_prompt, job_id, expect_array=False)

            # Store the table text for validation
            self._table_text = structured_text

            # Combine results
            if isinstance(items_data, list):
                metadata["invoice_data"] = self.fix_net_gross_confusion(items_data)
            elif isinstance(items_data, dict) and "invoice_data" in items_data:
                metadata["invoice_data"] = self.fix_net_gross_confusion(items_data["invoice_data"])
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

    def fix_net_gross_confusion(self, items: list) -> list:
        """
        Fix items where LLM confused net and gross columns
        If gross < net, they're likely swapped - auto-swap them
        """
        import re

        fixed_items = []
        swap_count = 0

        for item in items:
            if not isinstance(item, dict):
                fixed_items.append(item)
                continue

            try:
                # Extract numeric values
                net_str = re.sub(r'[^\d,.-]', '', str(item.get("net", "")))
                gross_str = re.sub(r'[^\d,.-]', '', str(item.get("gross", "")))

                if not net_str or not gross_str:
                    fixed_items.append(item)
                    continue

                # Convert to float
                net_val = float(net_str.replace(',', '.'))
                gross_val = float(gross_str.replace(',', '.'))

                # Check if they need swapping (gross should be > net, unless it's a discount/negative)
                if gross_val < net_val and gross_val > 0 and net_val > 0:
                    # Auto-swap the values
                    original_gross = item.get("gross")
                    original_net = item.get("net")
                    item["gross"] = original_net
                    item["net"] = original_gross
                    swap_count += 1
                    logger.info(f"Auto-swapped gross/net for item '{item.get('name', '')[:30]}': net {gross_val}→{net_val}, gross {net_val}→{gross_val}")

            except Exception as e:
                logger.debug(f"Could not validate net/gross for item: {e}")

            fixed_items.append(item)

        if swap_count > 0:
            logger.info(f"Fixed {swap_count} items with gross<net confusion by auto-swapping values")

        return fixed_items

    def validate_and_fix_quantities(self, items: List[Dict], table_text: str) -> List[Dict]:
        """
        Validate quantity extraction and fix obvious errors (Priority 2 fix)

        If qty=1 but unit_price != net, check if unit_price * qty = net would work with a different qty

        Args:
            items: List of invoice items
            table_text: Structured OCR text for reference

        Returns:
            Items with corrected quantities
        """
        fixed_items = []
        fix_count = 0

        for item in items:
            try:
                qty = float(item.get('quantity', '1').replace(',', '.')) if item.get('quantity') else 1.0
                unit_price_str = item.get('unit_price', '0')
                net_str = item.get('net', '0')

                if not unit_price_str or not net_str:
                    fixed_items.append(item)
                    continue

                unit_price = float(unit_price_str.replace(',', '.'))
                net = float(net_str.replace(',', '.'))

                # If qty=1 and unit_price * qty != net, calculate implied quantity
                if abs(qty - 1.0) < 0.01 and abs(unit_price - net) > 1.0:
                    # Check if net / unit_price gives a whole number
                    if unit_price > 0:
                        implied_qty = net / unit_price
                        # If implied qty is close to a whole number (within 0.1)
                        if abs(implied_qty - round(implied_qty)) < 0.1 and round(implied_qty) > 1:
                            original_qty = item['quantity']
                            item['quantity'] = str(int(round(implied_qty)))
                            fix_count += 1
                            logger.info(f"Fixed quantity for '{item.get('name', '')[:40]}': {original_qty} → {item['quantity']} (calculated from net={net}/unit_price={unit_price})")

            except (ValueError, ZeroDivisionError, TypeError) as e:
                logger.debug(f"Could not validate quantity for item: {e}")

            fixed_items.append(item)

        if fix_count > 0:
            logger.info(f"Fixed {fix_count} items with incorrect quantities")

        return fixed_items

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
                # First clean items
                cleaned_items = [
                    self.clean_invoice_item(item)
                    for item in data["invoice_data"]
                    if isinstance(item, dict)
                ]

                # Fix quantity errors (Priority 2)
                table_text = getattr(self, '_table_text', '')
                cleaned_items = self.validate_and_fix_quantities(cleaned_items, table_text)

                # Then validate/fix with VAT-aware price mapper
                # Use table text if available for better remapping
                table_text = getattr(self, '_table_text', '')
                table_lines = self._prepare_table_lines(table_text)
                row_matches = self._match_items_to_table_rows(cleaned_items, table_lines)

                validated_items = []
                for idx, item in enumerate(cleaned_items):
                    row_text = row_matches[idx] if idx < len(row_matches) else ''
                    validated_item = price_mapper.validate_and_fix_item(item, row_text)
                    validated_items.append(validated_item)

                data['invoice_data'] = validated_items

                # Validate sum consistency
                validation = self.validate_sum_consistency(data["invoice_data"])
                if validation["warnings"]:
                    for warning in validation["warnings"]:
                        logger.warning(f"Sum consistency: {warning}")

                # Add validation metadata
                if "_processing_metadata" not in data:
                    data["_processing_metadata"] = {}
                data["_processing_metadata"]["validation"] = {
                    "calculated_net_total": validation["total_net"],
                    "calculated_gross_total": validation["total_gross"],
                    "warnings": validation["warnings"]
                }

            # Auto-detect and set currency if missing
            if not data.get("currency") or data.get("currency") == "":
                # Check invoice_data for currency
                if "invoice_data" in data and data["invoice_data"]:
                    for item in data["invoice_data"]:
                        if item.get("currency"):
                            data["currency"] = item["currency"]
                            break
                # Default to HUF if still not found
                if not data.get("currency"):
                    data["currency"] = "HUF"

            return data

        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            return self.get_error_fallback(f"Post-processing error: {str(e)}")


    def _prepare_table_lines(self, table_text: str) -> List[str]:
        if not table_text:
            return []
        return [line.strip() for line in table_text.splitlines() if line.strip()]

    def _match_items_to_table_rows(self, items: List[Dict[str, Any]], table_lines: List[str]) -> List[str]:
        if not items:
            return []
        if not table_lines:
            return [""] * len(items)

        matches: List[str] = []
        used_indices: set = set()

        for item in items:
            idx = self._find_best_table_line_index(item, table_lines, used_indices)
            if idx is None:
                matches.append("")
            else:
                used_indices.add(idx)
                matches.append(table_lines[idx])
        return matches

    def _find_best_table_line_index(self, item: Dict[str, Any], table_lines: List[str], used_indices: set) -> Optional[int]:
        best_idx = None
        best_score = float('-inf')

        for idx, line in enumerate(table_lines):
            score = self._score_table_line(item, line)
            if idx in used_indices:
                score -= 0.5  # Prefer unused lines but allow reuse
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None or best_score <= 0:
            return None
        return best_idx

    def _score_table_line(self, item: Dict[str, Any], line: str) -> float:
        if not line:
            return float('-inf')

        score = 0.0
        candidates = price_mapper.extract_price_candidates(line)
        line_numbers = [abs(num) for num in candidates.get('prices', []) if isinstance(num, (int, float))]
        item_numbers = [abs(num) for num in self._extract_item_numbers(item)]

        match_count = 0
        for target in item_numbers:
            if target == 0:
                continue
            for num in line_numbers:
                if abs(num - target) <= 1.0:
                    match_count += 1
                    break
        score += match_count * 5

        line_normalized = self._normalize_text(line)
        item_tokens = self._extract_name_tokens(item.get('name', ''))
        if item_tokens:
            token_hits = sum(1 for token in item_tokens if token in line_normalized)
            score += token_hits * 1.5

        quantity = item.get('quantity')
        try:
            quantity_val = float(str(quantity).replace(',', '.')) if quantity not in (None, '') else None
        except ValueError:
            quantity_val = None
        if quantity_val is not None:
            if any(abs(num - abs(quantity_val)) <= 0.01 for num in line_numbers):
                score += 1

        summary_keywords = ('osszesen', 'total', 'subtotal', 'balance', 'due', 'befizetett', 'vegosszeg')
        if any(keyword in line_normalized for keyword in summary_keywords):
            score -= 3

        return score

    def _extract_item_numbers(self, item: Dict[str, Any]) -> List[float]:
        numbers: List[float] = []
        for key in ('unit_price', 'net', 'gross'):
            value = item.get(key)
            if value in (None, ''):
                continue
            try:
                numbers.append(float(str(value).replace(',', '.')))
            except ValueError:
                continue
        return numbers

    def _extract_name_tokens(self, name: str) -> List[str]:
        if not name:
            return []
        normalized = self._normalize_text(name)
        return [token for token in re.findall(r'[a-z0-9]+', normalized) if len(token) >= 3]

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ''
        normalized = unicodedata.normalize('NFKD', text)
        return ''.join(ch for ch in normalized if not unicodedata.combining(ch)).lower()

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

    def validate_unit_price(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate unit_price makes sense given quantity and net (Priority 3 fix)

        Detects cases where unit_price has an extra leading digit (OCR error)
        Example: "773.69" should be "73.69" if qty=4 and net=294.76

        Args:
            item: Invoice item dictionary

        Returns:
            Item with corrected unit_price if needed
        """
        try:
            qty_str = item.get('quantity', '')
            unit_price_str = item.get('unit_price', '')
            net_str = item.get('net', '')

            if not qty_str or not unit_price_str or not net_str:
                return item

            qty = float(qty_str.replace(',', '.'))
            unit_price = float(unit_price_str.replace(',', '.'))
            net = float(net_str.replace(',', '.'))

            # Expected: unit_price * quantity ≈ net
            expected_net = unit_price * qty

            # Check if there's a >10% difference
            if abs(expected_net - net) > net * 0.1:
                # Try removing first digit from unit_price
                unit_price_str_clean = str(unit_price_str).replace(',', '.').strip()

                # Only try if unit_price has at least 2 digits before decimal
                if '.' in unit_price_str_clean:
                    parts = unit_price_str_clean.split('.')
                    if len(parts[0]) >= 2:  # At least 2 digits before decimal
                        corrected_price_str = parts[0][1:] + '.' + parts[1]
                        try:
                            corrected_price = float(corrected_price_str)
                            # Check if corrected price matches net better
                            if abs(corrected_price * qty - net) < 1.0:
                                item['unit_price'] = corrected_price_str
                                logger.info(f"Fixed unit_price for '{item.get('name', '')[:40]}': {unit_price} → {corrected_price} (extra leading digit removed)")
                        except ValueError:
                            pass

        except (ValueError, IndexError, TypeError) as e:
            logger.debug(f"Could not validate unit_price: {e}")

        return item

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

        # Validate unit price (Priority 3 fix)
        cleaned_item = self.validate_unit_price(cleaned_item)

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

            # Remove thousand separators (dots or spaces before comma)
            import re
            # Remove dots used as thousand separators (e.g., 1.234,56 → 1234,56)
            cleaned = re.sub(r'\.(?=\d{3})', '', cleaned)
            # Remove spaces used as thousand separators (e.g., 1 234,56 → 1234,56)
            cleaned = re.sub(r'\s(?=\d{3})', '', cleaned)

            # Convert comma decimal separator to period
            if "," in cleaned and "." not in cleaned:
                cleaned = cleaned.replace(",", ".")
            elif "," in cleaned and "." in cleaned:
                # European format: 1.234,56 → 1234.56
                cleaned = cleaned.replace(".", "").replace(",", ".")

            # Remove any remaining non-numeric characters except periods and minus
            cleaned = re.sub(r'[^\d.-]', '', cleaned)

            return cleaned

        except Exception as e:
            logger.warning(f"Failed to clean numeric value '{value}': {str(e)}")
            return ""

    def validate_sum_consistency(self, items: list) -> Dict[str, Any]:
        """
        Validate that line items sum up correctly

        Args:
            items: List of invoice line items

        Returns:
            Dict with validation results and warnings
        """
        try:
            total_net = 0.0
            total_gross = 0.0
            warnings = []

            for i, item in enumerate(items):
                try:
                    if item.get("net"):
                        net_val = float(item["net"])
                        total_net += net_val
                    if item.get("gross"):
                        gross_val = float(item["gross"])
                        total_gross += gross_val

                    # Validate individual item: gross should be >= net (unless discount)
                    if item.get("net") and item.get("gross"):
                        net_val = float(item["net"])
                        gross_val = float(item["gross"])
                        if gross_val < net_val and gross_val > 0 and net_val > 0:
                            warnings.append(f"Item {i+1} '{item.get('name', '')[:30]}': gross ({gross_val}) < net ({net_val})")

                except (ValueError, TypeError) as e:
                    warnings.append(f"Item {i+1}: Invalid numeric values")

            return {
                "total_net": round(total_net, 2),
                "total_gross": round(total_gross, 2),
                "warnings": warnings
            }

        except Exception as e:
            logger.warning(f"Sum consistency validation failed: {str(e)}")
            return {"total_net": 0, "total_gross": 0, "warnings": [str(e)]}

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
