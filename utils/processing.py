import os
import logging
import uuid
import re
import unicodedata
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from .ocr_tesseract import TesseractOCRProcessor
from .llm import OllamaClient
from .progress import progress_tracker
from config import Config

logger = logging.getLogger(__name__)

class InvoiceProcessor:
    """Main processing pipeline that combines OCR and LLM"""

    def _prepare_text_for_llm(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return text
        text = self._normalize_numeric_tokens(text)
        text = self._normalize_dates(text)
        return text

    def _trim_metadata_text(self, text: str) -> str:
        """
        Extract metadata-relevant portion of OCR text.

        Strategy: Keep text BEFORE table headers + some lines after (in case seller/buyer info is at the bottom)
        For invoices where seller/buyer appears after the table, we need to capture both beginning and end.
        """
        if not text:
            return text

        lines = [line for line in text.splitlines()]
        header_keywords = (
            "megnevez",
            "cikkleir",
            "mennyiseg",
            "quantity",
            "description",
            "items",
            "termek",
            "listaar",
            "netto",
            "brutto",
        )

        # Find table header position
        table_start_idx = None
        for idx, line in enumerate(lines):
            normalized = self._normalize_text(line)
            if any(keyword in normalized for keyword in header_keywords):
                table_start_idx = idx
                break

        if table_start_idx is not None:
            # Strategy: Take text BEFORE table + check if seller/buyer keywords exist AFTER table
            before_table = lines[:table_start_idx]
            after_table = lines[table_start_idx:]

            # Check if seller/buyer keywords exist after the table (some invoices have this layout)
            seller_buyer_keywords = ("kft", "zrt", "bt", "vevo", "szallito", "elado", "szolgaltato", "adoszam")
            has_metadata_after = False
            for line in after_table[:50]:  # Check up to 50 lines after table
                normalized = self._normalize_text(line)
                if any(keyword in normalized for keyword in seller_buyer_keywords):
                    has_metadata_after = True
                    break

            if has_metadata_after:
                # Include text both before AND after table (up to reasonable limit)
                combined = before_table + after_table[:50]
                trimmed = [ln for ln in combined if ln.strip()]
                if trimmed:
                    return "\n".join(trimmed)
            else:
                # Just use text before table
                trimmed = [ln for ln in before_table if ln.strip()]
                if trimmed:
                    return "\n".join(trimmed)

        # No table headers found, return full text (up to reasonable limit for metadata)
        max_lines = 100  # Reasonable limit for metadata section
        trimmed = [ln for ln in lines[:max_lines] if ln.strip()]
        return "\n".join(trimmed)

    def _normalize_numeric_tokens(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            token = match.group(0)
            cleaned = token.replace("\u00a0", " ")

            # Remove spaces that act as thousand separators (digit + space + three digits + separator/end)
            cleaned = re.sub(r'(?<=\d)[ ](?=\d{3}(?:[^\d]|$))', '', cleaned)

            # Convert European decimal comma to dot
            cleaned = cleaned.replace(',', '.')
            return cleaned

        pattern = re.compile(r'-?\d[\d\s\u00a0]*(?:[.,]\d+)?')
        return pattern.sub(repl, text)

    def _normalize_dates(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            year, month, day = match.groups()
            return f"{year}.{int(month):02d}.{int(day):02d}"

        text = re.sub(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', repl, text)
        text = re.sub(r'(\d{4})\.(\d{1,2})\.(\d{1,2})', repl, text)
        return text

    def __init__(self, config: Config = None):
        self.config = config or Config()

        # Choose OCR engine based on configuration (currently only Tesseract)
        ocr_engine = getattr(config, 'OCR_ENGINE', 'tesseract').lower()
        if ocr_engine != 'tesseract':
            logger.warning("Configured OCR engine '%s' is not available; falling back to Tesseract", ocr_engine)
        logger.info("Using Tesseract OCR engine")
        self.ocr_processor = TesseractOCRProcessor(config)

        self.llm_client = OllamaClient(config)
        self._metadata_source_text: Optional[str] = None
        self._buyer_fallback: Optional[Dict[str, str]] = None
        self.table_extractor = None

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

    def _extract_pdf_native_text(self, pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Try to extract text directly from the PDF text layer.

        Returns:
            Dict with extracted text when a searchable PDF is detected, otherwise None.
        """
        try:
            import io
            import pypdfium2 as pdfium
        except Exception as exc:
            logger.debug(f"Native PDF text extraction unavailable: {exc}")
            return None

        start_time = datetime.utcnow()

        try:
            pdf_doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
        except Exception as exc:
            logger.debug(f"Failed to open PDF with pypdfium2: {exc}")
            return None

        texts: List[str] = []
        non_space_chars = 0
        digit_count = 0

        try:
            for page_index in range(len(pdf_doc)):
                try:
                    page = pdf_doc[page_index]
                    text_page = page.get_textpage()
                    page_text = text_page.get_text_range()
                except Exception as exc:
                    logger.debug(f"Native text extraction failed on page {page_index}: {exc}")
                    continue
                finally:
                    try:
                        text_page.close()
                    except Exception:
                        pass
                    try:
                        page.close()
                    except Exception:
                        pass

                if page_text:
                    cleaned = page_text.replace("\r", "")
                    non_space_chars += sum(1 for c in cleaned if not c.isspace())
                    digit_count += sum(1 for c in cleaned if c.isdigit())
                    texts.append(cleaned.strip())
        finally:
            try:
                pdf_doc.close()
            except Exception:
                pass

        if not texts:
            return None

        # Require a substantial amount of text to avoid false positives on scanned PDFs
        if non_space_chars < 400 and digit_count < 60:
            return None

        joined_text = "\n\n=== PAGE BREAK ===\n\n".join(texts)
        duration = (datetime.utcnow() - start_time).total_seconds()
        density = non_space_chars / max(len(pdf_bytes), 1)

        return {
            "plain_text": joined_text,
            "structured_text": joined_text,
            "char_count": non_space_chars,
            "digit_count": digit_count,
            "text_density": density,
            "duration": duration
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

            # Step 1: Extract text using OCR or native PDF text (5% - 20%)
            logger.info("Step 1: OCR text extraction...")
            if job_id:
                progress_tracker.update_progress(job_id, "ocr", 8, "Starting OCR processing", "processing")

            use_native_text = False
            native_text = self._extract_pdf_native_text(pdf_bytes)
            structured_text_metadata: Optional[str] = None
            structured_text_for_items: Optional[str] = None

            if native_text:
                use_native_text = True
                extracted_text = self._prepare_text_for_llm(native_text["plain_text"])
                structured_text_metadata = self._prepare_text_for_llm(native_text["structured_text"])
                structured_text_for_items = structured_text_metadata
                ocr_duration = native_text["duration"]
                logger.info(
                    "Detected searchable PDF text layer (chars=%s, digits=%s, density=%.4f). Using native text.",
                    native_text["char_count"],
                    native_text["digit_count"],
                    native_text["text_density"],
                )
            else:
                ocr_start = datetime.utcnow()
                extracted_text = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id, preserve_columns=False)
                extracted_text = self._prepare_text_for_llm(extracted_text)
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

            self._buyer_fallback = None
            # Chunk 1: Extract metadata (use structured OCR for better layout preservation - Priority 1 fix)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 30, "Extracting metadata with structured OCR", "processing")

            # Use structured OCR for metadata to preserve two-column layouts and field boundaries
            if not use_native_text or structured_text_metadata is None:
                structured_text_metadata = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id, preserve_columns=True)
                structured_text_metadata = self._prepare_text_for_llm(structured_text_metadata)
                structured_text_metadata = self._trim_metadata_text(structured_text_metadata)
                logger.info(f"Using structured OCR for metadata extraction ({len(structured_text_metadata)} chars)")
            else:
                # Already have native text - just trim it
                structured_text_metadata = self._trim_metadata_text(structured_text_metadata)
                logger.info(f"Using native PDF text for metadata extraction ({len(structured_text_metadata)} chars)")

            self._metadata_source_text = structured_text_metadata
            metadata_prompt = self.llm_client.create_extraction_prompt(structured_text_metadata, "metadata")
            metadata = self.llm_client.generate_completion(metadata_prompt, job_id)
            metadata = self._seed_buyer_from_text(metadata, structured_text_metadata)
            buyer_snapshot = metadata.get("buyer")
            if isinstance(buyer_snapshot, dict):
                self._buyer_fallback = buyer_snapshot.copy()

            # Chunk 2: Extract line items (use structured OCR for table)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 60, "Extracting line items with structured OCR", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during chunked processing")
                    return self.get_cancellation_result(filename, job_id)

            # Re-OCR with table structure for items extraction
            structured_text, used_doctr = self._extract_item_table_text(
                pdf_bytes,
                job_id,
                structured_text_for_items,
                use_native_text,
            )
            if used_doctr:
                logger.info(f"Using docTR table extractor for items ({len(structured_text)} chars)")
            elif use_native_text and structured_text_for_items is not None:
                logger.info(f"Using native PDF text for item extraction ({len(structured_text)} chars)")
            else:
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
            metadata_source = getattr(self, "_metadata_source_text", None) or extracted_text
            final_data = self._seed_buyer_from_text(final_data, metadata_source)

            buyer_dict = final_data.setdefault("buyer", {})
            if isinstance(self._buyer_fallback, dict):
                self._merge_buyer_info(buyer_dict, self._buyer_fallback)

            raw_buyer = self._extract_buyer_from_text(metadata_source)
            if raw_buyer:
                self._merge_buyer_info(buyer_dict, raw_buyer)

            self._sanitize_buyer_fields(buyer_dict)
            self._metadata_source_text = None
            self._buyer_fallback = None

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

    def remove_duplicate_items(self, items: List[Dict]) -> List[Dict]:
        """
        Remove duplicate items that appear multiple times in extraction.

        Duplicates occur when items appear on multiple pages (e.g., last items on page 2
        repeat on page 3) and OCR extracts both occurrences.

        Strategy: Track seen items by signature (normalized name + quantity + net)
        and keep only the first occurrence.

        Args:
            items: List of invoice items

        Returns:
            Deduplicated list of items
        """
        if not items:
            return items

        seen = {}
        unique_items = []
        duplicate_count = 0

        for i, item in enumerate(items):
            # Create signature: normalized name + quantity + net
            name = item.get('name', '').strip().lower()
            qty = item.get('quantity', '')
            net = item.get('net', '')

            # Create unique signature
            sig = f"{name}|{qty}|{net}"

            if sig in seen:
                # Duplicate found
                duplicate_count += 1
                first_idx = seen[sig]
                logger.info(f"Removing duplicate item #{i+1} (duplicate of item #{first_idx+1}): '{item.get('name', '')[:50]}' (qty={qty}, net={net})")
            else:
                # First occurrence - keep it
                seen[sig] = len(unique_items)
                unique_items.append(item)

        if duplicate_count > 0:
            logger.info(f"Removed {duplicate_count} duplicate items (from {len(items)} to {len(unique_items)} items)")

        return unique_items

    def remove_header_section(self, ocr_text: str) -> str:
        """
        Remove header section from OCR text to prevent LLM from extracting wrong data.
        The header typically contains company branding and wrong email addresses.

        Strategy: Find where the actual invoice content starts by looking for markers like:
        - "Vevő:" or "Vev6:" (Buyer section)
        - Date fields like "Teljesítés dátuma" (Fulfillment date)
        - Table headers like "Cikkleiras" (Item description)

        Args:
            ocr_text: Raw OCR text

        Returns:
            OCR text with header removed (starting from buyer section)
        """
        if not ocr_text:
            return ocr_text

        lines = ocr_text.split('\n')

        # Find the line where actual invoice content starts
        # Look for buyer section markers (most reliable)
        buyer_markers = ['Vevő:', 'Vev6:', 'VEVŐ:', 'Buyer:', 'Customer:']

        for i, line in enumerate(lines):
            # Check for buyer section
            for marker in buyer_markers:
                if marker in line:
                    logger.info(f"Removing header: Found buyer marker '{marker}' at line {i}")
                    # Return from this line onwards
                    return '\n'.join(lines[i:])

        # Fallback: Look for date field markers (Teljesítés, Fizetési, etc.)
        date_markers = ['Teljesités', 'Teljesítés', 'Fizetési', 'Bizonylat kelte']
        for i, line in enumerate(lines):
            for marker in date_markers:
                if marker in line and i > 5:  # Must be after first few lines
                    logger.info(f"Removing header: Found date marker '{marker}' at line {i}")
                    # Go back a few lines to include buyer section
                    start_line = max(0, i - 3)
                    return '\n'.join(lines[start_line:])

        # If no markers found, return original (better safe than sorry)
        logger.warning("Could not find header markers, keeping full OCR text")
        return ocr_text

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

            # Format dates consistently (YYYY.MM.DD with dots)
            for date_field in ["issue_date", "due_date", "fulfillment_date"]:
                if date_field in data and data[date_field]:
                    data[date_field] = self.normalize_date_string(data[date_field])

            # Normalize tax IDs to short form (country code + number)
            if "seller" in data and isinstance(data["seller"], dict):
                tax_id = data["seller"].get("tax_id", "")
                if tax_id:
                    data["seller"]["tax_id"] = self.normalize_tax_id(tax_id)

            if "buyer" in data and isinstance(data["buyer"], dict):
                tax_id = data["buyer"].get("tax_id", "")
                if tax_id:
                    data["buyer"]["tax_id"] = self.normalize_tax_id(tax_id)

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

                # Remove duplicate items (same name + quantity + net)
                cleaned_items = self.remove_duplicate_items(cleaned_items)

                data['invoice_data'] = cleaned_items

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
                        if isinstance(item, dict) and item.get("currency"):
                            data["currency"] = item["currency"]
                            break
            # Default to HUF if still not found
            if not data.get("currency"):
                data["currency"] = "HUF"

            # Ensure each line item inherits invoice currency when missing
            invoice_currency = data.get("currency")
            if invoice_currency and "invoice_data" in data and isinstance(data["invoice_data"], list):
                for item in data["invoice_data"]:
                    if isinstance(item, dict) and not item.get("currency"):
                        item["currency"] = invoice_currency

            return data

        except Exception as e:
            logger.error(f"Post-processing failed: {str(e)}")
            return self.get_error_fallback(f"Post-processing error: {str(e)}")


    def _prepare_table_lines(self, table_text: str) -> List[str]:
        if not table_text:
            return []
        normalized_text = self._normalize_table_text(table_text)
        return [line.strip() for line in normalized_text.splitlines() if line.strip()]

    def _normalize_table_text(self, text: str) -> str:
        if not text:
            return text
        merged_lines = self._merge_orphan_numeric_lines(text.splitlines())
        rows = self._parse_table_rows(merged_lines)
        if rows:
            return "\n".join(" | ".join(row) for row in rows)
        cleaned = [line.strip() for line in merged_lines if line.strip()]
        return "\n".join(cleaned)

    def _extract_item_table_text(
        self,
        pdf_bytes: bytes,
        job_id: Optional[str],
        native_structured_text: Optional[str],
        use_native_text: bool,
    ) -> Tuple[str, bool]:
        raw_text: Optional[str] = None
        used_doctr = False

        if self.table_extractor and (not use_native_text or native_structured_text is None):
            try:
                doctr_text = self.table_extractor.extract_table_text(pdf_bytes, job_id)
                if doctr_text and doctr_text.strip():
                    raw_text = doctr_text
                    used_doctr = True
            except Exception as exc:
                logger.warning("docTR table extractor failed, falling back to OCR processor: %s", exc)

        if raw_text is None:
            if not use_native_text or native_structured_text is None:
                raw_text = self.ocr_processor.extract_text_from_pdf(
                    pdf_bytes,
                    job_id,
                    preserve_columns=True,
                )
            else:
                raw_text = native_structured_text

        prepared = self._prepare_text_for_llm(raw_text or "")
        normalized = self._normalize_table_text(prepared)
        return normalized, used_doctr

    def _merge_orphan_numeric_lines(self, lines: List[str]) -> List[str]:
        if not lines:
            return []

        merged: List[str] = []
        for line in lines:
            stripped = line.rstrip()
            if not stripped:
                merged.append(stripped)
                continue

            if merged and self._is_numeric_tail(stripped):
                merged[-1] = (merged[-1] + " " + stripped).strip()
            else:
                merged.append(stripped)
        return merged

    def _is_numeric_tail(self, line: str) -> bool:
        if not line:
            return False

        stripped = line.strip()
        if not stripped:
            return False

        if not any(ch.isdigit() for ch in stripped):
            return False

        letters = ''.join(ch for ch in stripped if ch.isalpha())
        letters_lower = letters.lower()
        for token in ("ft", "huf", "eur", "usd", "gbp"):
            letters_lower = letters_lower.replace(token, "")

        return letters_lower == ""

    def _parse_table_rows(self, lines: List[str]) -> List[List[str]]:
        header_keywords = (
            "megnevez",
            "cikkleir",
            "description",
            "mennyiseg",
            "quantity",
            "unit",
            "netto",
            "brutto",
        )
        stop_keywords = (
            "oldal",
            "page",
            "==",
            "osszesen",
            "osszegzes",
            "payment",
            "fizetesi",
            "seller",
            "szallitas",
            "szamla",
        )

        rows: List[List[str]] = []
        buffer: List[str] = []
        header_seen = False
        pattern = r'-?\d[\d\s ]*(?:[.,]\d+)?'

        for raw_line in lines:
            stripped = raw_line.strip()
            if not stripped:
                continue

            normalized = self._normalize_text(stripped)
            if not header_seen and any(keyword in normalized for keyword in header_keywords):
                header_seen = True
                buffer.clear()
                continue

            if not header_seen:
                continue

            if any(keyword in normalized for keyword in header_keywords):
                buffer.clear()
                continue

            if any(stop in normalized for stop in stop_keywords):
                break

            buffer.append(stripped)
            candidate = " ".join(buffer)
            matches = list(re.finditer(pattern, candidate))
            valid_matches = []
            for match in matches:
                start_pos = match.start()
                end_pos = match.end()
                before = candidate[start_pos - 1] if start_pos > 0 else " "
                after = candidate[end_pos] if end_pos < len(candidate) else " "
                if before.isalpha() or after.isalpha():
                    continue
                valid_matches.append(match)

            if len(valid_matches) < 4:
                continue

            first_match = valid_matches[0]
            name = candidate[:first_match.start()].strip(" |:-")
            if not name:
                continue

            numbers = [self._clean_numeric_token(m.group(0)) for m in valid_matches]
            quantity = numbers[0]
            unit_price = numbers[1]
            net = numbers[-2]
            gross = numbers[-1]
            rows.append([name, quantity, unit_price, net, gross])
            buffer.clear()

        return rows

    def _clean_numeric_token(self, token: str) -> str:
        cleaned = token.replace(' ', '').replace(' ', '')
        cleaned = cleaned.replace(',', '.')
        if cleaned.count('.') > 1:
            parts = cleaned.split('.')
            cleaned = parts[0] + '.' + ''.join(parts[1:])
        return cleaned

    def _seed_buyer_from_text(self, metadata: Any, source_text: Optional[str]) -> Any:
        if not isinstance(metadata, dict) or not source_text:
            return metadata

        buyer = metadata.get("buyer")
        if not isinstance(buyer, dict):
            buyer = {}
            metadata["buyer"] = buyer

        candidate = self._extract_buyer_from_text(source_text)
        if candidate:
            self._merge_buyer_info(buyer, candidate)
        return metadata

    def _merge_buyer_info(self, target: Dict[str, str], candidate: Dict[str, str]) -> None:
        if not candidate:
            return

        company_tokens = ("kft", "zrt", "bt", "nyrt", "kht")

        for key, value in candidate.items():
            if not value:
                continue
            existing = target.get(key, "")

            if not existing:
                target[key] = value
                continue

            if key == "name":
                cand_has_company = any(token in value.lower() for token in company_tokens)
                existing_has_company = any(token in existing.lower() for token in company_tokens)
                if cand_has_company and not existing_has_company:
                    target[key] = value
                continue

            if key == "address":
                if len(value) > len(existing):
                    target[key] = value
                continue

            if key == "tax_id" and existing != value:
                target[key] = value

    def _sanitize_buyer_fields(self, buyer: Dict[str, str]) -> None:
        if not isinstance(buyer, dict):
            return

        contact_tokens = (
            "@",
            "email",
            "telefon",
            "tel",
            "order",
            "rendeles",
            "rendelés",
            "azonosito",
            "azonosító",
            "szamla szam",
            "számla szám",
        )

        name = buyer.get("name")
        if name and any(token in name.lower() for token in contact_tokens):
            buyer["name"] = ""

        address = buyer.get("address")
        if address and any(token in address.lower() for token in contact_tokens):
            buyer["address"] = ""

    def _extract_buyer_from_text(self, text: str) -> Optional[Dict[str, str]]:
        if not text:
            return None

        lines = [line.strip() for line in text.splitlines()]
        normalized = [self._normalize_text(line) for line in lines]

        def marker_present(norm_line: str) -> bool:
            letters_only = re.sub(r'[^a-z]', '', norm_line)
            for marker in primary_markers + secondary_markers:
                compact = marker.replace(" ", "")
                if marker in norm_line or compact in letters_only:
                    return True
            return False

        primary_markers = [
            "vevo",
            "vev",
            "megrendelo",
            "ugyfel",
            "szamla cimzettje",
            "szamlacimzettje",
            "szamlafizeto",
            "fizeto",
            "szamla befogadoja",
            "elofizeto",
            "vasarlo",
        ]
        secondary_markers = [
            "szallitas cim",
            "szallitasi cim",
            "szallitasicim",
        ]

        segment = self._locate_anchor_section(lines, normalized, primary_markers)
        if segment is None:
            segment = self._locate_anchor_section(lines, normalized, secondary_markers)

        if not segment:
            return None

        marker_index = next((i for i, norm in enumerate(normalized) if marker_present(norm)), -1)

        skip_keywords = (
            "adoszam",
            "addeszam",
            "tax",
            "bank",
            "bankszamla",
            "iban",
            "swift",
            "telefon",
            "tel",
            "email",
            "fax",
            "kapcsolat",
            "contact",
        )

        name = ""
        address_parts: List[str] = []
        tax_id = ""

        for raw_line in segment:
            stripped = raw_line.strip().strip(":")
            if not stripped:
                continue

            normalized_line = self._normalize_text(stripped)

            if not tax_id:
                matches = re.findall(r'[A-Z]{2}\s?\d{6,}|[0-9]{8}-\d-[0-9]{2}|[0-9]{11}', stripped.replace(" ", ""))
                for match in matches:
                    normalized_tax = self.normalize_tax_id(match)
                    if normalized_tax:
                        tax_id = normalized_tax
                        break

            if not name and any(term in normalized_line for term in ("adoszam", "addeszam")):
                company_candidate = self._extract_company_name_from_tax_line(stripped, tax_id)
                if company_candidate:
                    name = company_candidate

            if any(keyword in normalized_line for keyword in skip_keywords):
                continue
            if not any(ch.isalpha() for ch in stripped):
                continue
            if stripped[0].isdigit():
                address_parts.append(stripped)
                continue

            if not name:
                name = stripped
            else:
                address_parts.append(stripped)

        result: Dict[str, str] = {}
        if not name and tax_id:
            for line in lines:
                if tax_id in line:
                    candidate_name = self._extract_company_name_from_tax_line(line, tax_id)
                    if candidate_name:
                        name = candidate_name
                        break
        if not name and marker_index != -1:
            company_tokens = ("kft", "zrt", "bt", "nyrt", "kht")
            for idx in range(marker_index + 1, min(len(lines), marker_index + 8)):
                line = lines[idx].strip()
                if not line:
                    continue
                norm_line = normalized[idx]
                if any(token in norm_line for token in company_tokens):
                    name = line.rstrip(":")
                    break
        if not name:
            for line in lines:
                norm_line = self._normalize_text(line)
                if any(token in norm_line for token in ("kft", "zrt", "bt", "nyrt", "kht")) and len(line) <= 80:
                    name = line.strip().rstrip(":")
                    break
        if name:
            result["name"] = name
        if address_parts:
            result["address"] = ', '.join(address_parts)
        if tax_id:
            result["tax_id"] = tax_id

        return result if result else None

    def _extract_company_name_from_tax_line(self, line: str, tax_id: Optional[str]) -> Optional[str]:
        working_line = line
        if tax_id:
            for token in {tax_id, tax_id.replace('-', ''), tax_id.replace('-', '').replace('HU', '')}:
                if token:
                    working_line = working_line.replace(token, '')

        segments = working_line.split('/')
        tail = segments[-1].strip(" :-") if segments else working_line
        if not tail:
            tail = working_line.strip(" :-")
        tail = re.sub(r'^[A-Z]{1,3}\b', '', tail).strip()

        company_pattern = re.compile(
            r'([A-Z0-9][A-Za-z0-9\-\.\u00C0-\u024F\s]{1,80}?(?:KFT\.?|ZRT\.?|BT\.?|KHT\.?|NYRT\.?))',
            re.IGNORECASE
        )
        match = None
        for candidate in company_pattern.finditer(tail):
            match = candidate
        if match:
            return match.group(1).strip().rstrip(":")
        return None

    def _locate_anchor_section(
        self,
        lines: List[str],
        normalized_lines: List[str],
        markers: List[str]
    ) -> Optional[List[str]]:
        if not markers:
            return None

        stop_keywords = (
            "bank",
            "bankszamla",
            "iban",
            "swift",
            "telefon",
            "tel",
            "email",
            "fax",
            "kapcsolat",
            "contact",
            "payment",
            "fizetesi",
        )

        marker_pairs = [(marker, marker.replace(" ", "")) for marker in markers]

        def contains_marker(normalized_line: str) -> bool:
            letters_only = re.sub(r'[^a-z]', '', normalized_line)
            for marker, marker_compact in marker_pairs:
                if marker in normalized_line or marker_compact in letters_only:
                    return True
            return False

        for idx, norm in enumerate(normalized_lines):
            if contains_marker(norm):
                collected: List[str] = []
                current_line = lines[idx]

                # Include immediate previous descriptive line if marker stands alone
                if ":" not in current_line and idx > 0:
                    prev = lines[idx - 1].strip()
                    if prev and not contains_marker(normalized_lines[idx - 1]):
                        collected.append(prev)

                if ":" in current_line:
                    before, after = current_line.split(":", 1)
                    after_colon = after.strip()
                    if not after_colon and idx > 0:
                        prev = lines[idx - 1].strip()
                        if prev and not contains_marker(normalized_lines[idx - 1]):
                            collected.append(prev)
                    if after_colon:
                        collected.append(after_colon)

                j = idx + 1
                while j < len(lines):
                    candidate = lines[j].strip()
                    norm_candidate = normalized_lines[j]

                    if not candidate:
                        break
                    if contains_marker(norm_candidate):
                        break
                    if any(stop in norm_candidate for stop in stop_keywords):
                        break

                    collected.append(candidate)
                    if len(collected) >= 6:
                        break
                    j += 1

                collected = [entry for entry in collected if entry]
                if collected:
                    return collected

        return None

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
        # Extract numbers from line for matching
        import re
        line_numbers = [float(m.replace(',', '.')) for m in re.findall(r'\d+[.,]\d+', line)]
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

        # CRITICAL: Detect and fix barcode/massive number confusion
        cleaned_item = self.detect_barcode_confusion(cleaned_item)

        # Validate unit price (Priority 3 fix)
        cleaned_item = self.validate_unit_price(cleaned_item)

        return cleaned_item

    def detect_barcode_confusion(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect when LLM extracted barcodes/product codes as prices

        Common patterns:
        - quantity > 10000 (likely extracted barcode as quantity)
        - unit_price > 1000000 (likely extracted barcode as price)
        - net or gross > 100000000 (barcode confusion)

        Args:
            item: Invoice item dictionary

        Returns:
            Item with barcode confusion fixed (set invalid fields to empty)
        """
        try:
            # Check quantity for barcode confusion
            if item.get("quantity"):
                try:
                    qty = float(str(item["quantity"]).replace(',', '.'))
                    if qty > 10000:
                        logger.warning(f"Item '{item.get('name', '')[:30]}': quantity {qty} too large (likely barcode) - clearing")
                        item["quantity"] = ""
                except ValueError:
                    pass

            # Check unit_price for barcode confusion
            if item.get("unit_price"):
                try:
                    unit_price = float(str(item["unit_price"]).replace(',', '.'))
                    if unit_price > 1000000:
                        logger.warning(f"Item '{item.get('name', '')[:30]}': unit_price {unit_price} too large (likely barcode) - clearing")
                        item["unit_price"] = ""
                except ValueError:
                    pass

            # Check net for massive number confusion
            if item.get("net"):
                try:
                    net = float(str(item["net"]).replace(',', '.'))
                    if net > 100000000:
                        logger.warning(f"Item '{item.get('name', '')[:30]}': net {net} too large (likely barcode) - clearing")
                        item["net"] = ""
                except ValueError:
                    pass

            # Check gross for massive number confusion
            if item.get("gross"):
                try:
                    gross = float(str(item["gross"]).replace(',', '.'))
                    if gross > 100000000:
                        logger.warning(f"Item '{item.get('name', '')[:30]}': gross {gross} too large (likely barcode) - clearing")
                        item["gross"] = ""
                except ValueError:
                    pass

        except Exception as e:
            logger.debug(f"Could not detect barcode confusion: {e}")

        return item

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

    def normalize_date_string(self, value: Any) -> str:
        """
        Normalize a date-like string to YYYY.MM.DD with padded month/day.
        Falls back to the cleaned string if parsing fails.
        """
        if not value:
            return ""

        try:
            raw = str(value).strip()
            if not raw:
                return ""

            raw = unicodedata.normalize("NFKD", raw)
            raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
            raw = raw.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", "-")

            # Try to infer order from digit groups first
            parts = re.findall(r'\d+', raw)
            if len(parts) >= 3:
                if len(parts[0]) == 4:
                    year, month, day = parts[0], parts[1], parts[2]
                elif len(parts[2]) == 4:
                    year, month, day = parts[2], parts[1], parts[0]
                else:
                    year, month, day = parts[0], parts[1], parts[2]

                parsed = datetime(int(year), int(month), int(day))
                return parsed.strftime("%Y.%m.%d")

            sanitized = re.sub(r'[^0-9./-]', '', raw).strip(".")
            candidate_formats = [
                "%Y.%m.%d",
                "%Y-%m-%d",
                "%Y/%m/%d",
                "%d.%m.%Y",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%m.%d.%Y",
                "%m-%d-%Y",
                "%m/%d/%Y",
            ]

            for fmt in candidate_formats:
                try:
                    parsed = datetime.strptime(sanitized, fmt)
                    return parsed.strftime("%Y.%m.%d")
                except ValueError:
                    continue

            # Last fallback: replace separators and trim trailing dots
            fallback = sanitized.replace("-", ".").replace("/", ".")
            fallback = re.sub(r'\.+', '.', fallback).strip(".")
            return fallback

        except Exception as exc:
            logger.debug(f"Failed to normalize date '{value}': {exc}")
            sanitized = str(value).replace("-", ".").replace("/", ".")
            return sanitized.strip().strip(".")

    def normalize_tax_id(self, tax_id: str) -> str:
        """
        Normalize tax ID to Hungarian format: XXXXXXXX-X-XX

        Examples:
            "24144094-2-20/HU24144094" -> "24144094-2-20"
            "12337545.02.13" -> "12337545-2-13" (fix dot separator and leading zero)
            "SK 2022210311" -> "SK2022210311"
            "HU24144094" -> "HU24144094"

        Args:
            tax_id: Raw tax ID string

        Returns:
            Normalized tax ID (Hungarian: XXXXXXXX-X-XX or country code + number)
        """
        if not tax_id or tax_id.strip() == "":
            return ""

        original_value = str(tax_id).strip()
        value = original_value
        value = unicodedata.normalize("NFKD", value)
        value = "".join(ch for ch in value if not unicodedata.combining(ch))
        value = value.replace("\u00a0", " ")
        value = value.replace("\u2013", "-").replace("\u2014", "-")

        # Remove common labels
        value = re.sub(r'(?i)adoszam[:\s-]*', '', value)
        value = re.sub(r'(?i)vat[:\s-]*', '', value)
        value = re.sub(r'(?i)tax[\s-]?id[:\s-]*', '', value)

        # Split on "/" and look for explicit country-code formats (e.g., "XXXXXXXX-X-XX/HUXXXXXXXX")
        segments = [segment.strip() for segment in value.split("/") if segment.strip()]
        for segment in segments:
            compact = segment.upper().replace(" ", "")
            if re.match(r'^[A-Z]{2}\d+$', compact):
                logger.debug(f"Tax ID normalization: '{original_value}' -> '{compact}' (country code format)")
                return compact

        value = value.strip()
        upper_value = value.upper()

        # Collapse whitespace between CC and digits (e.g., "SK 2022210311")
        cc_match = re.match(r'^([A-Z]{2})[\s-]*(\d+)$', upper_value)
        if cc_match:
            result = f"{cc_match.group(1)}{cc_match.group(2)}"
            logger.debug(f"Tax ID normalization: '{original_value}' -> '{result}' (CC format)")
            return result

        # CRITICAL FIX: Handle Hungarian tax ID with dots (e.g., "12337545.02.13")
        # Pattern: XXXXXXXX.XX.XX or XXXXXXXX.X.XX
        dot_pattern = re.match(r'^(\d{8})\.(\d{1,2})\.(\d{2})$', value)
        if dot_pattern:
            part1, part2, part3 = dot_pattern.groups()
            # Remove leading zero from middle part (e.g., "02" -> "2")
            part2 = str(int(part2))
            result = f"{part1}-{part2}-{part3}"
            logger.debug(f"Tax ID normalization: '{original_value}' -> '{result}' (dot format)")
            return result

        # Handle dashed Hungarian pattern that may have leading zero
        dash_pattern = re.match(r'^(\d{8})-(\d{1,2})-(\d{2})$', value)
        if dash_pattern:
            part1, part2, part3 = dash_pattern.groups()
            # Remove leading zero from middle part if present
            part2 = str(int(part2))
            result = f"{part1}-{part2}-{part3}"
            logger.debug(f"Tax ID normalization: '{original_value}' -> '{result}' (dash format)")
            return result

        # Extract all digits
        digits_only = re.sub(r'\D', '', upper_value)

        # Handle Hungarian pattern (8-1-2 digits). Drop spurious leading zero in middle segment if present.
        if len(digits_only) == 12 and digits_only[8] == "0":
            digits_only = digits_only[:8] + digits_only[9:]

        if len(digits_only) >= 11:
            core = digits_only[:11]
            try:
                # Format as XXXXXXXX-X-XX (standard Hungarian format)
                result = f"{core[:8]}-{core[8]}-{core[9:]}"
                logger.debug(f"Tax ID normalization: '{original_value}' -> '{result}' (digits only format)")
                return result
            except Exception:
                pass

        if digits_only:
            logger.debug(f"Tax ID normalization: '{original_value}' -> '{digits_only}' (fallback digits)")
            return digits_only

        # Fallback: clean and normalize separators
        fallback = re.sub(r'[^A-Z0-9-]', '', upper_value)
        fallback = re.sub(r'-{2,}', '-', fallback).strip('-')
        logger.debug(f"Tax ID normalization: '{original_value}' -> '{fallback}' (final fallback)")
        return fallback

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
