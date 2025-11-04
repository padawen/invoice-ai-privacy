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

    def _split_text_by_columns(self, text: str) -> tuple[str, str]:
        """
        Split text into left and right columns for seller/buyer separation.

        Uses hybrid approach:
        1. Try separator-based split (if │ character exists)
        2. Fallback to label-based extraction (search for SZÁLLÍTÓ/VEVŐ)

        Args:
            text: Full invoice text with potential two-column layout

        Returns:
            (left_column_text, right_column_text)
            Left column typically contains seller, right contains buyer
        """
        if not text:
            return text, ""

        import re

        # Strategy 1: Separator-based split (fast, for native PDFs)
        # Look for box-drawing characters: │ ┬ ├ ┤
        if '│' in text or '┬' in text:
            logger.info("Using separator-based column split (found │ character)")
            lines = text.split('\n')
            left_lines = []
            right_lines = []

            for line in lines:
                if '│' in line:
                    parts = line.split('│', 1)  # Split only on first occurrence
                    left_lines.append(parts[0].strip())
                    if len(parts) > 1:
                        right_lines.append(parts[1].strip())
                elif '┬' in line:
                    # Header separator line - skip
                    continue
                else:
                    # Line without separator - include in both (likely metadata)
                    left_lines.append(line)

            left_text = '\n'.join(left_lines).strip()
            right_text = '\n'.join(right_lines).strip()

            if left_text and right_text:
                logger.info(f"Separator split successful: left={len(left_text)} chars, right={len(right_text)} chars")
                return left_text, right_text

        # Strategy 2: Label-based extraction (robust, for all formats)
        # Find SZÁLLÍTÓ/VEVŐ labels and extract associated sections
        logger.info("Using label-based column extraction (no separator found)")

        # Find seller section (SZÁLLÍTÓ, Szállító, Seller)
        seller_patterns = [
            r'(?:─+\s*)?SZ[ÁA]LL[ÍI]T[ÓO]\s*(?:─+)?',  # SZÁLLÍTÓ with optional separators
            r'Sz[áa]ll[íi]t[óo]\s*:?',                   # Szállító with optional colon
            r'Seller\s*:?',                               # English
        ]

        # Find buyer section (VEVŐ, Vevő, Buyer, Customer)
        buyer_patterns = [
            r'(?:─+\s*)?VEV[ŐO]\s*(?:─+)?',             # VEVŐ with optional separators
            r'Vev[őo]\s*:?',                              # Vevő with optional colon
            r'Buyer\s*:?',                                # English
            r'Customer\s*:?',                             # English alt
        ]

        seller_start = -1
        buyer_start = -1

        # Find seller start
        for pattern in seller_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                seller_start = match.end()
                logger.debug(f"Found seller section at position {seller_start}")
                break

        # Find buyer start
        for pattern in buyer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                buyer_start = match.end()
                logger.debug(f"Found buyer section at position {buyer_start}")
                break

        # Extract sections
        left_text = ""
        right_text = ""

        if seller_start > 0:
            # Extract ~300 chars after seller label (enough for name, address, tax ID)
            seller_end = min(seller_start + 300, len(text))
            # Or until we hit buyer section
            if buyer_start > seller_start:
                seller_end = min(seller_end, buyer_start)
            left_text = text[seller_start:seller_end].strip()

        if buyer_start > 0:
            # Extract ~300 chars after buyer label
            buyer_end = min(buyer_start + 300, len(text))
            # Stop at next major section (items table, dates, etc.)
            # Look for section markers
            next_section_patterns = [
                r'Cikksz[áa]m',  # Items table
                r'Sz[áa]mla kelte',  # Invoice date
                r'T[ée]telek',  # Items summary
            ]
            for pattern in next_section_patterns:
                match = re.search(pattern, text[buyer_start:buyer_end], re.IGNORECASE)
                if match:
                    buyer_end = buyer_start + match.start()
                    break
            right_text = text[buyer_start:buyer_end].strip()

        if left_text or right_text:
            logger.info(f"Label-based extraction successful: left={len(left_text)} chars, right={len(right_text)} chars")
            return left_text, right_text

        # Fallback: return full text for both (no column detection possible)
        logger.warning("Column split failed - returning full text for both sides")
        return text, text

    def _extract_items_section(self, text: str) -> str:
        """
        Extract only the items table section from text.

        This removes header/footer/metadata to help LLM focus on actual line items.
        Works for both OCR and native PDF text.

        Args:
            text: Full invoice text

        Returns:
            Items section only, or full text if section not found
        """
        if not text:
            return text

        import re

        # Find table start - look for common table headers
        table_start_patterns = [
            r'Cikksz[aá]m.*?Megnevez[eé]s',  # Hungarian: Cikkszám Megnevezés
            r'Megnevez[eé]s.*?Mennyis[eé]g',  # Megnevezés Mennyiség
            r'Description.*?Quantity',         # English
            r'Item.*?Qty',                     # English short
        ]

        start_pos = -1
        for pattern in table_start_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                start_pos = match.start()
                logger.debug(f"Found items table start at position {start_pos} with pattern: {pattern}")
                break

        if start_pos == -1:
            # No table header found, return full text
            logger.debug("No items table header found, returning full text")
            return text

        # Find table end - look for summary/total lines
        table_end_patterns = [
            r'T[eé]telek [oö]sszesen',        # Tételek összesen
            r'[oö]sszesen:',                   # Összesen:
            r'V[eé]g[oö]sszeg',                # Végösszeg
            r'Total:',                         # English
            r'Subtotal:',                      # English
            r'Fizet[eé]s',                     # Fizetés/Fizetendő
            r'Azaz:',                          # Azaz: (amount in words)
        ]

        # Start searching from table start position
        search_text = text[start_pos:]
        end_pos = len(text)  # Default: end of text

        for pattern in table_end_patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                # Found end marker, position is relative to start_pos
                end_pos = start_pos + match.start()
                logger.debug(f"Found items table end at position {end_pos} with pattern: {pattern}")
                break

        # Extract section
        items_section = text[start_pos:end_pos].strip()

        logger.info(f"Extracted items section: {len(items_section)} chars (from {len(text)} total)")

        return items_section

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
            logger.info("Step 2: LLM structure extraction (separate entity extraction strategy)...")
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 25, "Starting LLM processing", "processing")

            llm_start = datetime.utcnow()

            # New 4-chunk strategy - separate seller, buyer, metadata, items
            logger.info(f"Extracting invoice data in 4 chunks (separate seller/buyer extraction)")

            # UNIVERSAL SPATIAL COLUMN DETECTION for all PDFs
            if not use_native_text or structured_text_metadata is None:
                # OCR PATH: Use Tesseract bbox-based column detection
                logger.info("Using Tesseract bbox-based column detection (OCR path)")
                left_column_text, right_column_text, structured_text_metadata = self.ocr_processor.extract_text_with_columns(pdf_bytes, job_id)
                left_column_text = self._prepare_text_for_llm(left_column_text)
                right_column_text = self._prepare_text_for_llm(right_column_text)
                structured_text_metadata = self._prepare_text_for_llm(structured_text_metadata)
                logger.info(f"OCR spatial split: left={len(left_column_text)} chars (seller), right={len(right_column_text)} chars (buyer), full={len(structured_text_metadata)} chars")
            else:
                # NATIVE PDF PATH: Use PyMuPDF bbox-based column detection
                logger.info("Using PyMuPDF bbox-based column detection (native text path)")
                left_column_text, right_column_text, structured_text_metadata = self.ocr_processor.extract_text_with_columns_native(pdf_bytes, job_id)
                left_column_text = self._prepare_text_for_llm(left_column_text)
                right_column_text = self._prepare_text_for_llm(right_column_text)
                structured_text_metadata = self._prepare_text_for_llm(structured_text_metadata)
                logger.info(f"Native spatial split: left={len(left_column_text)} chars (seller), right={len(right_column_text)} chars (buyer), full={len(structured_text_metadata)} chars")

            # Chunk 1: Extract seller (25% -> 35%)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 30, "Extracting seller information", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during seller extraction")
                    return self.get_cancellation_result(filename, job_id)

            # Use LEFT column for seller extraction (seller is always on the left)
            seller_prompt = self.llm_client.create_extraction_prompt(left_column_text, "seller")
            seller_data = self.llm_client.generate_completion(seller_prompt, job_id)
            logger.info(f"Seller extraction completed: {seller_data.get('seller', {}).get('name', 'N/A')}")

            # Chunk 2: Extract buyer (35% -> 45%)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 40, "Extracting buyer information", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during buyer extraction")
                    return self.get_cancellation_result(filename, job_id)

            # Use RIGHT column for buyer extraction (buyer is always on the right)
            buyer_prompt = self.llm_client.create_extraction_prompt(right_column_text, "buyer")
            buyer_data = self.llm_client.generate_completion(buyer_prompt, job_id)

            # Fallback: If buyer extraction failed (returns just label or empty), retry with full text
            buyer_name = buyer_data.get('buyer', {}).get('name', '').strip()
            is_label_only = buyer_name.upper() in ['VEVŐ', 'VEVO', 'BUYER', 'CUSTOMER', 'VEV6', 'VÁSÁRLÓ', 'MEGRENDELŐ']
            is_empty = not buyer_name or len(buyer_name) < 5

            if is_label_only or is_empty:
                logger.warning(f"⚠️  Buyer extraction failed from right column (got: '{buyer_name}'), retrying with full text")
                buyer_prompt_fallback = self.llm_client.create_extraction_prompt(structured_text_metadata, "buyer")
                buyer_data_fallback = self.llm_client.generate_completion(buyer_prompt_fallback, job_id)

                # Validate fallback result before using it
                fallback_name = buyer_data_fallback.get('buyer', {}).get('name', '').strip()
                is_fallback_valid = (
                    fallback_name and
                    len(fallback_name) >= 5 and
                    fallback_name.upper() not in ['VEVŐ', 'VEVO', 'BUYER', 'CUSTOMER', 'VEV6']
                )

                if is_fallback_valid:
                    buyer_data = buyer_data_fallback
                    logger.info(f"✓ Fallback buyer extraction succeeded: {fallback_name}")
                else:
                    logger.warning(f"✗ Fallback also failed: {fallback_name}")

            logger.info(f"Buyer extraction completed: {buyer_data.get('buyer', {}).get('name', 'N/A')}")

            # Chunk 3: Extract invoice metadata (dates, numbers) (45% -> 55%)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 50, "Extracting invoice metadata", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during metadata extraction")
                    return self.get_cancellation_result(filename, job_id)

            metadata_prompt = self.llm_client.create_extraction_prompt(structured_text_metadata, "metadata")
            invoice_metadata = self.llm_client.generate_completion(metadata_prompt, job_id)
            logger.info(f"Invoice metadata extraction completed: {invoice_metadata.get('invoice_number', 'N/A')}")

            # Merge seller, buyer, and metadata into single dict
            metadata = {
                "seller": seller_data.get("seller", {}),
                "buyer": buyer_data.get("buyer", {}),
                "invoice_number": invoice_metadata.get("invoice_number", ""),
                "issue_date": invoice_metadata.get("issue_date", ""),
                "fulfillment_date": invoice_metadata.get("fulfillment_date", ""),
                "due_date": invoice_metadata.get("due_date", ""),
                "payment_method": invoice_metadata.get("payment_method", ""),
                "currency": invoice_metadata.get("currency", "")
            }

            # Chunk 4: Extract line items (55% -> 90%)
            if job_id:
                progress_tracker.update_progress(job_id, "llm", 60, "Extracting line items", "processing")
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled during items extraction")
                    return self.get_cancellation_result(filename, job_id)

            # Use structured OCR for items - hybrid approach: try preserve_columns=True first
            # If that yields 0 items, retry with preserve_columns=False
            if not use_native_text or structured_text_for_items is None:
                # First attempt: preserve_columns=True (generally better for most invoices)
                structured_text_for_items = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id, preserve_columns=True)
                structured_text_for_items = self._prepare_text_for_llm(structured_text_for_items)
                logger.info(f"Using structured OCR for items extraction with preserve_columns=True ({len(structured_text_for_items)} chars)")
            else:
                logger.info(f"Using native PDF text for item extraction ({len(structured_text_for_items)} chars)")
                # CRITICAL FIX: Extract only items section from native text
                # Native text contains full document (seller, buyer, headers, footers)
                # which confuses LLM - we need to isolate the items table
                structured_text_for_items = self._extract_items_section(structured_text_for_items)
                logger.info(f"After items section extraction: {len(structured_text_for_items)} chars")

            items_prompt = self.llm_client.create_extraction_prompt(structured_text_for_items, "items")
            items_data = self.llm_client.generate_completion(items_prompt, job_id, expect_array=False)

            # Extract items from response
            extracted_items = []
            if isinstance(items_data, list):
                extracted_items = items_data
            elif isinstance(items_data, dict) and "invoice_data" in items_data:
                extracted_items = items_data["invoice_data"]

            logger.info(f"[HYBRID DEBUG] First attempt (preserve_columns=True) extracted {len(extracted_items)} items")

            # Hybrid fallback: if 0 items extracted with preserve_columns=True, retry with False
            # But ONLY for OCR-based PDFs - never fallback for native text PDFs
            should_retry = len(extracted_items) == 0 and not use_native_text
            logger.info(f"[HYBRID DEBUG] Should retry: {should_retry} (items={len(extracted_items)}, use_native_text={use_native_text})")

            if should_retry:
                logger.warning("No items extracted with preserve_columns=True, retrying with preserve_columns=False")
                structured_text_for_items_retry = self.ocr_processor.extract_text_from_pdf(pdf_bytes, job_id, preserve_columns=False)
                structured_text_for_items_retry = self._prepare_text_for_llm(structured_text_for_items_retry)
                logger.info(f"Retrying items extraction with preserve_columns=False ({len(structured_text_for_items_retry)} chars)")

                items_prompt_retry = self.llm_client.create_extraction_prompt(structured_text_for_items_retry, "items")
                items_data_retry = self.llm_client.generate_completion(items_prompt_retry, job_id, expect_array=False)

                if isinstance(items_data_retry, list):
                    extracted_items = items_data_retry
                elif isinstance(items_data_retry, dict) and "invoice_data" in items_data_retry:
                    extracted_items = items_data_retry["invoice_data"]

                logger.info(f"[HYBRID DEBUG] Retry with preserve_columns=False yielded {len(extracted_items)} items")
                if len(extracted_items) > 0:
                    logger.info(f"[HYBRID DEBUG] ✓ Retry SUCCESS - fallback to preserve_columns=False fixed the issue")
                else:
                    logger.warning(f"[HYBRID DEBUG] ✗ Retry FAILED - still 0 items with preserve_columns=False")

            # Apply post-processing and combine results
            metadata["invoice_data"] = self.fix_net_gross_confusion(extracted_items) if extracted_items else []

            structured_data = metadata

            # Normalize date formats (2024.12.04 → 2024-12-04)
            structured_data = self._normalize_dates(structured_data)

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

    def _normalize_dates(self, structured_data: dict) -> dict:
        """
        Normalize date formats in structured data.
        Converts dates from formats like "2024.12.04", "2025.01.09." to "2024-12-04"

        Applies to: issue_date, fulfillment_date, due_date
        """
        import re

        date_fields = ["issue_date", "fulfillment_date", "due_date"]

        for field in date_fields:
            if field not in structured_data or not structured_data[field]:
                continue

            date_str = str(structured_data[field]).strip()

            # Pattern: YYYY.MM.DD or YYYY.MM.DD. (with optional trailing dot)
            match = re.match(r'(\d{4})\.(\d{1,2})\.(\d{1,2})\.?', date_str)
            if match:
                year, month, day = match.groups()
                normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                structured_data[field] = normalized
                logger.debug(f"Normalized {field}: {date_str} → {normalized}")
                continue

            # Pattern: DD.MM.YYYY or DD.MM.YYYY. (European format)
            match = re.match(r'(\d{1,2})\.(\d{1,2})\.(\d{4})\.?', date_str)
            if match:
                day, month, year = match.groups()
                normalized = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                structured_data[field] = normalized
                logger.debug(f"Normalized {field}: {date_str} → {normalized}")

        return structured_data

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
