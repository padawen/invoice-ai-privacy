"""
Tesseract-based OCR processor for invoice text extraction
Alternative to docTR with better table detection
"""

import logging
import io
from typing import BinaryIO
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF for native PDF text with bboxes
import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class TesseractOCRProcessor:
    """OCR processor using Tesseract engine"""

    def __init__(self, config=None):
        """
        Initialize Tesseract OCR processor

        Args:
            config: Configuration object with OCR settings
        """
        self.config = config
        self.dpi = config.OCR_DPI if config else 300

        # Tesseract configuration
        # --psm 6: Assume a single uniform block of text (good for invoices)
        # --oem 3: Use both legacy and LSTM OCR engines
        self.tesseract_config = r'--psm 6 --oem 3'

        logger.info(f"TesseractOCRProcessor initialized (DPI: {self.dpi})")

    def extract_text_from_pdf(self, pdf_file, job_id: str = None, preserve_columns: bool = False) -> str:
        """
        Extract text from PDF using Tesseract OCR

        Args:
            pdf_file: Binary file object or bytes containing PDF data
            job_id: Optional job ID for progress tracking (unused but kept for API compatibility)
            preserve_columns: If True, attempts to preserve column layout

        Returns:
            Extracted text as string
        """
        try:
            logger.info(f"Starting Tesseract OCR extraction (preserve_columns={preserve_columns})")

            # Handle both bytes and file-like objects
            if isinstance(pdf_file, bytes):
                pdf_bytes = pdf_file
            elif hasattr(pdf_file, 'read'):
                pdf_bytes = pdf_file.read()
                pdf_file.seek(0)  # Reset file pointer
            else:
                raise ValueError("pdf_file must be bytes or a file-like object")

            # Convert PDF to images
            logger.info(f"Converting PDF to images at {self.dpi} DPI")
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt='png',
                thread_count=4
            )

            logger.info(f"Converted PDF to {len(images)} images at {self.dpi} DPI")

            # Extract text from each page
            full_text = []

            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}/{len(images)}")

                # Simple text extraction - TSV approach was too complex for LLM
                config = self.tesseract_config if not preserve_columns else r'--psm 1 --oem 3'

                # Extract text using Tesseract
                page_text = pytesseract.image_to_string(
                    image,
                    lang='hun+eng',  # Hungarian + English
                    config=config
                )

                full_text.append(page_text.strip())

                # Add page break marker
                if page_num < len(images):
                    full_text.append("\n=== PAGE BREAK ===\n")

            # Combine all pages
            extracted_text = "\n".join(full_text)

            # Post-process: Remove header email to prevent LLM from picking wrong one
            # The header shows "duovill@index.hu" but the correct contact email is "duovill@duovill.com"
            extracted_text = self._filter_header_email(extracted_text)

            # Post-process: Clean item lines to remove product codes and tax percentages
            # This makes price columns clearer for LLM
            extracted_text = self._clean_item_lines(extracted_text)

            logger.info(f"Extracted {len(extracted_text)} characters from {len(images)} pages")

            return extracted_text

        except Exception as e:
            logger.error(f"Tesseract OCR extraction failed: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")

    def extract_text_with_layout(self, pdf_file: BinaryIO) -> str:
        """
        Extract text from PDF preserving layout structure using TSV output
        This provides bounding boxes and confidence scores

        Args:
            pdf_file: Binary file object containing PDF data

        Returns:
            Extracted text with layout preserved
        """
        try:
            logger.info("Starting Tesseract OCR extraction with layout preservation")

            # Read PDF bytes
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)

            # Convert PDF to images
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt='png',
                thread_count=4
            )

            logger.info(f"Converted PDF to {len(images)} images")

            full_text = []

            for page_num, image in enumerate(images, 1):
                logger.info(f"Processing page {page_num}/{len(images)} with layout detection")

                # Get TSV output with bounding boxes
                tsv_data = pytesseract.image_to_data(
                    image,
                    lang='hun+eng',
                    config=r'--psm 6 --oem 3',
                    output_type=pytesseract.Output.DICT
                )

                # Reconstruct text preserving layout
                page_text = self._reconstruct_layout(tsv_data)
                full_text.append(page_text)

                if page_num < len(images):
                    full_text.append("\n=== PAGE BREAK ===\n")

            extracted_text = "\n".join(full_text)
            logger.info(f"Extracted {len(extracted_text)} characters with layout preservation")

            return extracted_text

        except Exception as e:
            logger.error(f"Layout extraction failed: {str(e)}")
            raise Exception(f"OCR layout processing failed: {str(e)}")

    def _reconstruct_layout(self, tsv_data: dict) -> str:
        """
        Reconstruct text layout from Tesseract TSV data
        Groups words by line based on vertical position

        Args:
            tsv_data: Dictionary from pytesseract.image_to_data

        Returns:
            Text with preserved layout
        """
        lines = {}

        # Group words by line number and block
        for i in range(len(tsv_data['text'])):
            text = tsv_data['text'][i].strip()
            if not text:
                continue

            conf = int(tsv_data['conf'][i])
            if conf < 30:  # Skip low confidence text
                continue

            block_num = tsv_data['block_num'][i]
            line_num = tsv_data['line_num'][i]
            left = tsv_data['left'][i]

            # Create unique line identifier
            line_key = (block_num, line_num)

            if line_key not in lines:
                lines[line_key] = []

            lines[line_key].append((left, text))

        # Sort lines by block and line number
        sorted_lines = sorted(lines.items())

        # Reconstruct text
        result = []
        for line_key, words in sorted_lines:
            # Sort words by horizontal position
            sorted_words = sorted(words, key=lambda x: x[0])
            line_text = ' '.join(word for _, word in sorted_words)
            result.append(line_text)

        return '\n'.join(result)

    def extract_text_with_columns(self, pdf_file, job_id: str = None) -> tuple[str, str, str]:
        """
        Extract text split into left/right columns using bbox coordinates.

        Universal approach that works for any language/layout by using spatial information.
        Detects columns based on X-coordinate positions.

        Args:
            pdf_file: Binary file object or bytes containing PDF data
            job_id: Optional job ID for progress tracking

        Returns:
            tuple: (left_column_text, right_column_text, full_text)
                   - left_column_text: Text from left side (typically seller)
                   - right_column_text: Text from right side (typically buyer)
                   - full_text: Complete text for fallback
        """
        try:
            logger.info("Starting bbox-based column detection")

            # Handle both bytes and file-like objects
            if isinstance(pdf_file, bytes):
                pdf_bytes = pdf_file
            elif hasattr(pdf_file, 'read'):
                pdf_bytes = pdf_file.read()
                if hasattr(pdf_file, 'seek'):
                    pdf_file.seek(0)
            else:
                raise ValueError("pdf_file must be bytes or a file-like object")

            # Convert PDF to images
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt='png',
                thread_count=4
            )

            logger.info(f"Converted PDF to {len(images)} images for column detection")

            # Process first page only (metadata usually on page 1)
            image = images[0]

            # Get TSV output with bounding boxes
            tsv_data = pytesseract.image_to_data(
                image,
                lang='hun+eng',
                config=r'--psm 1 --oem 3',  # Auto layout detection
                output_type=pytesseract.Output.DICT
            )

            # Split into columns based on X-coordinate
            left_text, right_text, full_text = self._split_by_bbox_columns(tsv_data)

            logger.info(f"Column split: left={len(left_text)} chars, right={len(right_text)} chars, full={len(full_text)} chars")

            return left_text, right_text, full_text

        except Exception as e:
            logger.error(f"Bbox-based column detection failed: {str(e)}")
            # Fallback: extract simple text and return as full text
            try:
                full_text = self.extract_text_from_pdf(pdf_file, job_id, preserve_columns=False)
                return full_text, full_text, full_text
            except:
                raise Exception(f"OCR column processing failed: {str(e)}")

    def _split_by_bbox_columns(self, tsv_data: dict, page_width: float = None) -> tuple[str, str, str]:
        """
        Split text into left/right columns based on bounding box X-coordinates.

        Args:
            tsv_data: Dictionary from pytesseract.image_to_data or converted format
            page_width: Optional page width (if known from PDF). If None, will be calculated from bbox data.

        Returns:
            tuple: (left_column_text, right_column_text, full_text)
        """
        # Collect all words with their positions
        words = []
        if page_width is None:
            page_width = 0

        for i in range(len(tsv_data['text'])):
            text = tsv_data['text'][i].strip()
            if not text:
                continue

            conf = int(tsv_data['conf'][i])
            if conf < 30:  # Skip low confidence text
                continue

            left = tsv_data['left'][i]
            top = tsv_data['top'][i]
            width = tsv_data['width'][i]
            height = tsv_data['height'][i]

            words.append({
                'text': text,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'right': left + width
            })

            # Track page width
            page_width = max(page_width, left + width)

        if not words:
            logger.warning("No words detected in bbox column split")
            return "", "", ""

        # Use clustering to detect natural column boundaries
        # Extract X-coordinates (left position) for clustering
        x_coords = np.array([[w['left']] for w in words])

        # Detect if text naturally forms 1 or 2 columns
        # Check X-coordinate distribution
        x_values = [w['left'] for w in words]
        x_range = max(x_values) - min(x_values)
        x_std = np.std(x_values)

        # If X-coordinates have low standard deviation, likely single-column
        if x_std < page_width * 0.15:
            logger.info(f"Single-column layout detected (x_std={x_std:.1f}, threshold={page_width * 0.15:.1f})")
            full_text = self._reconstruct_column(words)
            return full_text, full_text, full_text

        # Use k-means with k=2 to find natural column boundaries
        try:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(x_coords)
            cluster_centers = kmeans.cluster_centers_.flatten()

            # Sort cluster centers to identify left and right
            sorted_centers = sorted(cluster_centers)
            left_center, right_center = sorted_centers[0], sorted_centers[1]

            # Calculate boundary as midpoint between cluster centers
            boundary = (left_center + right_center) / 2

            logger.info(f"Clustering detected columns: left_center={left_center:.1f}, right_center={right_center:.1f}, boundary={boundary:.1f}")

            # Split words based on cluster assignment
            left_words = [w for i, w in enumerate(words) if cluster_labels[i] == np.argmin(cluster_centers)]
            right_words = [w for i, w in enumerate(words) if cluster_labels[i] == np.argmax(cluster_centers)]

            # Validation: Check if columns are reasonably balanced
            left_ratio = len(left_words) / len(words)
            right_ratio = len(right_words) / len(words)

            if left_ratio < 0.15 or right_ratio < 0.15:
                logger.warning(f"Column split very imbalanced (left={left_ratio:.1%}, right={right_ratio:.1%}), treating as single-column")
                full_text = self._reconstruct_column(words)
                return full_text, full_text, full_text

            logger.info(f"Split words: {len(left_words)} left ({left_ratio:.1%}), {len(right_words)} right ({right_ratio:.1%})")

        except Exception as e:
            logger.warning(f"Clustering failed: {e}, falling back to midpoint split")
            # Fallback to simple midpoint
            midpoint = page_width / 2
            left_words = [w for w in words if w['left'] < midpoint]
            right_words = [w for w in words if w['left'] >= midpoint]
            logger.info(f"Fallback midpoint split: {len(left_words)} left, {len(right_words)} right")

        # Reconstruct text for each column
        left_text = self._reconstruct_column(left_words)
        right_text = self._reconstruct_column(right_words)
        full_text = self._reconstruct_column(words)

        # If one column is empty or very small, might be single-column layout
        if len(left_text) < 50 or len(right_text) < 50:
            logger.warning(f"One column is very small (left={len(left_text)}, right={len(right_text)}), may be single-column layout")
            # Don't split - return full text for both
            return full_text, full_text, full_text

        # CRITICAL: Validate column assignment - check if columns are swapped
        left_text, right_text = self._validate_column_assignment(left_text, right_text)

        return left_text, right_text, full_text

    def _reconstruct_column(self, words_list: list) -> str:
        """
        Reconstruct text from a list of words with position info.
        Sorts by vertical position, groups into lines.

        Args:
            words_list: List of dicts with 'text', 'top', 'left' keys

        Returns:
            Reconstructed text with line breaks
        """
        if not words_list:
            return ""

        # Sort by vertical position (top), then horizontal (left)
        sorted_words = sorted(words_list, key=lambda w: (w['top'], w['left']))

        # Group into lines (words with similar Y-coordinate)
        lines = []
        current_line = []
        current_y = sorted_words[0]['top']

        for word in sorted_words:
            # If Y-coordinate differs by more than 10px, start new line
            if abs(word['top'] - current_y) > 10:
                if current_line:
                    lines.append(' '.join(w['text'] for w in current_line))
                current_line = [word]
                current_y = word['top']
            else:
                current_line.append(word)

        # Add last line
        if current_line:
            lines.append(' '.join(w['text'] for w in current_line))

        return '\n'.join(lines)

    def _validate_column_assignment(self, left_text: str, right_text: str) -> tuple[str, str]:
        """
        Validate that left column contains seller and right contains buyer.
        If swapped, auto-correct by swapping them back.

        This is critical for scanned invoices where OCR bbox detection may be inaccurate,
        causing k-means clustering to assign columns backwards.

        Args:
            left_text: Text from left column
            right_text: Text from right column

        Returns:
            tuple: (seller_text, buyer_text) in correct order
        """
        import re

        # Patterns for seller labels (appear in seller section)
        # Include common OCR errors: Ó→0, Á→4, Í→1, etc.
        seller_patterns = [
            r'SZ[ÁA4]LL[ÍI1]T[ÓO0]',  # SZÁLLÍTÓ (with OCR error tolerance)
            r'ELAD[ÓO0]',               # ELADÓ (Ó can be read as 0)
            r'Seller',
            r'Kiállító',
            r'Ki[áa]ll[íi]t[óo]',
            r'Eladó',
            r'Elad[óo]',
        ]

        # Patterns for buyer labels (appear in buyer section)
        # Include common OCR errors: Ő→6 or 0, Ö→6, Ó→0, etc.
        buyer_patterns = [
            r'VEV[ŐOÖ06]',             # VEVŐ (Ő often read as 6, 0, or Ö)
            r'Buyer',
            r'Customer',
            r'Vásárló',
            r'V[áa]s[áa]rl[óo]',
            r'Megrendel[őo6]',          # Megrendelő (ő can be 6 or o)
        ]

        # Check which labels appear in which column
        left_has_seller = any(re.search(p, left_text, re.IGNORECASE) for p in seller_patterns)
        left_has_buyer = any(re.search(p, left_text, re.IGNORECASE) for p in buyer_patterns)
        right_has_seller = any(re.search(p, right_text, re.IGNORECASE) for p in seller_patterns)
        right_has_buyer = any(re.search(p, right_text, re.IGNORECASE) for p in buyer_patterns)

        # Detect if columns are swapped
        # Expected: left has seller labels, right has buyer labels
        # Swapped: left has buyer labels, right has seller labels
        if right_has_seller and left_has_buyer and not left_has_seller:
            logger.warning("⚠️  DETECTED SWAPPED COLUMNS - Auto-correcting: swapping left↔right")
            logger.info(f"Left column had buyer label, right column had seller label - swapping back")
            return right_text, left_text  # Swap them back to correct order

        # Normal case: left = seller, right = buyer
        if left_has_seller and right_has_buyer:
            logger.info("✓ Column assignment validated: left=seller, right=buyer")
        elif not left_has_seller and not right_has_seller and not left_has_buyer and not right_has_buyer:
            logger.warning("No seller/buyer labels detected in either column - checking for vertical stacking...")

            # NEW: Detect vertically-stacked layout (both entities in same column)
            # Pattern: no labels found + left column much larger + multiple company names
            left_ratio = len(left_text) / max(len(left_text) + len(right_text), 1)

            # Count company indicators (Kft., Zrt., Bt., Ltd., Inc., GmbH)
            company_pattern = r'\b(Kft|Zrt|Bt|Nyrt|Ltd|Inc|GmbH|LLC)\b'
            left_companies = len(re.findall(company_pattern, left_text, re.IGNORECASE))
            right_companies = len(re.findall(company_pattern, right_text, re.IGNORECASE))

            logger.info(f"Column analysis: left_ratio={left_ratio:.2f}, left_companies={left_companies}, right_companies={right_companies}")

            if left_ratio > 0.50 and left_companies >= 2 and right_companies == 0:
                logger.warning(f"⚠️  DETECTED VERTICAL-STACKED LAYOUT: {left_companies} companies in left column")
                logger.info("Splitting left column vertically into seller (first) and buyer (second)...")
                seller_text, buyer_text = self._split_vertical_companies(left_text)
                logger.info(f"Vertical split result: seller={len(seller_text)} chars, buyer={len(buyer_text)} chars")
                return seller_text, buyer_text

        return left_text, right_text

    def _split_vertical_companies(self, text: str) -> tuple[str, str]:
        """
        Split text containing 2 companies stacked vertically into seller and buyer.

        Strategy: Find company name lines, extract surrounding context for each.

        Args:
            text: Text containing both seller and buyer companies

        Returns:
            tuple: (seller_text, buyer_text)
        """
        import re

        lines = text.split('\n')

        # Find all lines containing company suffixes
        company_pattern = r'.+?\s+(Kft|Zrt|Bt|Nyrt|Ltd|Inc|GmbH|LLC)\b'
        company_line_indices = []

        for i, line in enumerate(lines):
            if re.search(company_pattern, line, re.IGNORECASE):
                company_line_indices.append(i)
                logger.debug(f"Found company at line {i}: {line[:50]}")

        if len(company_line_indices) >= 2:
            # We found at least 2 companies
            first_company_idx = company_line_indices[0]
            second_company_idx = company_line_indices[1]

            logger.info(f"Company 1 at line {first_company_idx}, Company 2 at line {second_company_idx}")

            # Extract seller: from first company line, take ~6 lines (company + address + tax ID + phone)
            seller_start = first_company_idx
            seller_end = min(first_company_idx + 6, second_company_idx)

            # Extract buyer: from second company line, take ~6 lines
            buyer_start = second_company_idx
            buyer_end = min(second_company_idx + 6, len(lines))

            seller_lines = lines[seller_start:seller_end]
            buyer_lines = lines[buyer_start:buyer_end]

            seller_text = '\n'.join(seller_lines)
            buyer_text = '\n'.join(buyer_lines)

            logger.debug(f"Seller excerpt: {seller_text[:100]}")
            logger.debug(f"Buyer excerpt: {buyer_text[:100]}")

            return seller_text, buyer_text

        # Fallback: simple midpoint split if company detection fails
        logger.warning("Could not find 2 distinct companies, using midpoint split as fallback")
        mid = len(lines) // 2
        return '\n'.join(lines[:mid]), '\n'.join(lines[mid:])

    def extract_text_with_columns_native(self, pdf_file, job_id: str = None) -> tuple[str, str, str]:
        """
        Extract text with spatial column detection from NATIVE PDF text layer.
        Uses PyMuPDF to get text with bounding boxes directly from PDF (no OCR needed).
        Universal approach that works for any language/layout.

        Args:
            pdf_file: Binary file object or bytes containing PDF data
            job_id: Optional job ID for progress tracking

        Returns:
            tuple: (left_column_text, right_column_text, full_text)
        """
        try:
            logger.info("Extracting text with spatial columns from native PDF (PyMuPDF)")

            # Read PDF bytes
            if isinstance(pdf_file, bytes):
                pdf_bytes = pdf_file
            else:
                pdf_bytes = pdf_file.read()
                if hasattr(pdf_file, 'seek'):
                    pdf_file.seek(0)

            # Open PDF with PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Extract text with bboxes from first page
            page = doc[0]
            text_dict = page.get_text("dict")

            # Convert PyMuPDF format to Tesseract-like bbox format
            words = []
            page_width = page.rect.width

            for block in text_dict["blocks"]:
                if block["type"] != 0:  # Skip non-text blocks (images, etc.)
                    continue

                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue

                        bbox = span["bbox"]  # (x0, y0, x1, y1)

                        # Convert to Tesseract format: {text, left, top, width, height}
                        words.append({
                            'text': text,
                            'left': int(bbox[0]),
                            'top': int(bbox[1]),
                            'width': int(bbox[2] - bbox[0]),
                            'height': int(bbox[3] - bbox[1])
                        })

            doc.close()

            # Convert to Tesseract-like dict format for bbox splitting
            tsv_data = {
                'text': [w['text'] for w in words],
                'left': [w['left'] for w in words],
                'top': [w['top'] for w in words],
                'width': [w['width'] for w in words],
                'height': [w['height'] for w in words],
                'conf': [100] * len(words)  # Native text has perfect confidence
            }

            # Use existing bbox column splitting logic
            left_text, right_text, full_text = self._split_by_bbox_columns(tsv_data, page_width=page_width)

            logger.info(f"Native PDF spatial split: left={len(left_text)} chars, right={len(right_text)} chars")
            return left_text, right_text, full_text

        except Exception as e:
            logger.error(f"Error extracting native PDF with spatial columns: {str(e)}")
            # Fallback: extract without spatial splitting
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                full_text = ""
                for page in doc:
                    full_text += page.get_text()
                doc.close()
                return full_text, full_text, full_text
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return "", "", ""

    def extract_tables(self, pdf_file: BinaryIO) -> list:
        """
        Extract tables from PDF using Tesseract

        Args:
            pdf_file: Binary file object containing PDF data

        Returns:
            List of table data structures
        """
        try:
            logger.info("Extracting tables using Tesseract")

            # Read PDF bytes
            pdf_bytes = pdf_file.read()
            pdf_file.seek(0)

            # Convert PDF to images
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.dpi,
                fmt='png',
                thread_count=4
            )

            all_tables = []

            for page_num, image in enumerate(images, 1):
                logger.info(f"Detecting tables on page {page_num}/{len(images)}")

                # Use PSM 6 for table detection
                tsv_data = pytesseract.image_to_data(
                    image,
                    lang='hun+eng',
                    config=r'--psm 6 --oem 3',
                    output_type=pytesseract.Output.DICT
                )

                # Extract table structure from TSV data
                table_data = self._extract_table_structure(tsv_data)
                if table_data:
                    all_tables.append({
                        'page': page_num,
                        'data': table_data
                    })

            logger.info(f"Extracted {len(all_tables)} tables")
            return all_tables

        except Exception as e:
            logger.error(f"Table extraction failed: {str(e)}")
            return []

    def _extract_table_structure(self, tsv_data: dict) -> list:
        """
        Extract table structure from Tesseract TSV output

        Args:
            tsv_data: Dictionary from pytesseract.image_to_data

        Returns:
            List of table rows (each row is a list of cells)
        """
        rows = {}

        for i in range(len(tsv_data['text'])):
            text = tsv_data['text'][i].strip()
            if not text:
                continue

            conf = int(tsv_data['conf'][i])
            if conf < 30:
                continue

            block_num = tsv_data['block_num'][i]
            line_num = tsv_data['line_num'][i]
            left = tsv_data['left'][i]
            top = tsv_data['top'][i]

            row_key = (block_num, line_num)

            if row_key not in rows:
                rows[row_key] = []

            rows[row_key].append({
                'text': text,
                'left': left,
                'top': top,
                'confidence': conf
            })

        # Convert to list of rows
        table_rows = []
        for row_key in sorted(rows.keys()):
            cells = sorted(rows[row_key], key=lambda x: x['left'])
            row_texts = [cell['text'] for cell in cells]
            table_rows.append(row_texts)

        return table_rows if table_rows else None

    def _filter_header_email(self, text: str) -> str:
        """
        Remove header email line to prevent LLM confusion.

        On DuóVill invoices, the header shows "Email: duovill@index.hu"
        but the contact person section shows "Email: duovill@duovill.com"

        We remove lines containing "duovill@index.hu" so the LLM only sees
        the correct contact email.

        Args:
            text: Extracted OCR text

        Returns:
            Filtered text with header email removed
        """
        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            # Skip lines containing the header email
            if 'duovill@index.hu' in line.lower():
                logger.debug(f"Filtering out header email line: {line.strip()}")
                continue

            filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _clean_item_lines(self, text: str) -> str:
        """
        Convert multi-line invoice items into single-row table format with headers.

        Handles TWO patterns:

        PATTERN 1 - Multi-line (has quantity unit on first line):
        000826 8536 1,00 db 1457,10 10% 27% 1457,10
        Finder relé 40.52 12VDC 1619,00 393,42 1850,52

        PATTERN 2 - Single-row (NO quantity unit, already single line):
        PRIMA-162 2P+F gyv 16A fehér dugaj IP20 730,00 177,39 834,39

        AFTER (both patterns become table rows):
        Megnevezés | Mennyiség | Egységár | Nettó | Bruttó
        Finder relé 40.52 12VDC | 1.00 | 1457.10 | 1457.10 | 1850.52
        PRIMA-162 2P+F gyv 16A fehér dugaj IP20 | | 730.00 | 177.39 | 834.39

        Args:
            text: Extracted OCR text

        Returns:
            Reformatted text with items in table rows
        """
        import re

        lines = text.split('\n')
        output_lines = []
        i = 0
        items_section_started = False
        table_header_added = False

        # Pattern to detect if we're in items section (look for header row)
        in_items_section = False

        while i < len(lines):
            line = lines[i]

            # Detect items section start (header row with "Cikkleiras" or "Mennyiség")
            if re.search(r'Cikkleiras|Mennyis.g', line, re.IGNORECASE):
                in_items_section = True
                output_lines.append(line)  # Keep header
                i += 1
                continue

            # Stop at end markers
            if re.search(r'Oldal:|Osszesen:|V.gosszesen:', line, re.IGNORECASE):
                in_items_section = False
                output_lines.append(line)
                i += 1
                continue

            # Only process lines in items section
            if not in_items_section:
                output_lines.append(line)
                i += 1
                continue

            # PATTERN 1: Check if this line contains quantity units (multi-line item)
            unit_match = re.search(r'(\d+[,\.]\d+)\s+(db|m|kg|l|pcs|pc)\s', line, re.IGNORECASE)

            if unit_match:
                # Mark that we're in items section
                if not items_section_started:
                    items_section_started = True
                    # Add table header before first item
                    if not table_header_added:
                        output_lines.append("\nMegnevezés | Mennyiség | Egységár | Nettó | Bruttó")
                        table_header_added = True

                # Extract quantity and unit
                quantity = unit_match.group(1).replace(',', '.')
                unit = unit_match.group(2)

                # Remove product codes, classification codes, and tax percentages
                cleaned_line = re.sub(r'^\d{6}\s+', '', line)  # Remove 6-digit product code
                cleaned_line = re.sub(r'\b\d{4,10}\s+', '', cleaned_line, count=1)  # Remove classification code
                cleaned_line = re.sub(r'\s+\d+%', '', cleaned_line)  # Remove tax percentages

                # Extract all decimal numbers from the cleaned line
                # Match numbers with comma/dot decimal separator: 1,00 or 103,70 or 10370,00
                # Also match numbers with space thousands separator: 10 370,00
                # Use word boundaries to avoid matching across separate numbers
                numbers = re.findall(r'\b\d+(?:\s\d{3})*[,\.]\d+\b', cleaned_line)
                # Convert to proper decimal format (remove spaces, replace comma with dot)
                numbers = [n.replace(' ', '').replace(',', '.') for n in numbers]

                # Structure after cleaning: quantity unit_price net
                # numbers[0] = quantity (already extracted)
                # numbers[1] = unit price (Nettó egységár)
                # numbers[2] = net value (Nettó érték) - appears twice on original line
                unit_price = numbers[1] if len(numbers) >= 2 else ""
                net = numbers[-1] if len(numbers) >= 2 else ""

                # Look at next line(s) for product name and gross price
                product_name = ""
                gross = ""
                j = i + 1

                # Next line should be product name (text line with possible additional prices)
                if j < len(lines):
                    next_line = lines[j].strip()

                    # Check if next line is a stop marker (end of items section)
                    if re.search(r'Oldal:|Osszesen:|V.gosszesen:|Jelen|NGM', next_line, re.IGNORECASE):
                        # This is end of items, don't use it as product name
                        # Mark end of section and keep the line as-is
                        in_items_section = False
                        output_lines.append(line)  # Keep the unit line as-is too
                        i += 1
                        continue

                    # Extract all numbers from product name line (these are prices)
                    # Match numbers like: 1619,00 or 393,42 or 1850,52
                    name_line_numbers = re.findall(r'\d+[,\.]\d+', next_line)
                    # Convert to float for comparison
                    name_line_numbers_float = []
                    name_line_numbers_str = []
                    for n in name_line_numbers:
                        clean_num = n.replace(',', '.')
                        name_line_numbers_float.append(float(clean_num))
                        name_line_numbers_str.append(clean_num)

                    # Product name is everything before the first PRICE number
                    # Key: Product names can have numbers like "2x0,75mm2" but prices have decimals like "115,22"
                    if name_line_numbers:
                        # Find the first number that looks like a price (has decimal with 2 digits, value >= 1)
                        # Price format: XX,XX or XXX,XX or XXXX,XX (at least 1 digit before comma/dot)
                        # This excludes product specs like "0,75" which is < 1
                        first_price_match = re.search(r'(?<!\d)[1-9]\d*[,\.]\d{2}', next_line)
                        if first_price_match:
                            product_name = next_line[:first_price_match.start()].strip()
                        else:
                            # Fallback: use first number
                            first_num = name_line_numbers[0]
                            first_num_pos = next_line.find(first_num)
                            if first_num_pos > 0:
                                product_name = next_line[:first_num_pos].strip()
                            else:
                                product_name = next_line.strip()

                        # The LARGEST number is the gross (bruttó)
                        # Because: unit_price < net < gross
                        max_idx = name_line_numbers_float.index(max(name_line_numbers_float))
                        gross = name_line_numbers_str[max_idx]
                    else:
                        # No numbers on this line, product name is whole line
                        product_name = next_line.strip()
                        # Check next line for gross
                        if j + 1 < len(lines):
                            next_next_line = lines[j + 1].strip()
                            extra_numbers = re.findall(r'\d+[,\.]\d+', next_next_line)
                            if extra_numbers:
                                gross = extra_numbers[-1].replace(',', '.')
                                i = j + 1  # Skip both lines
                            else:
                                i = j
                        else:
                            i = j

                    # Always skip the next line since we processed it
                    i = j

                # Format as table row
                table_row = f"{product_name} | {quantity} | {unit_price} | {net} | {gross}"
                output_lines.append(table_row)

                logger.debug(f"Converted PATTERN 1 to table row: {table_row}")

            # PATTERN 2: Single-row format (no quantity unit, already single line)
            # Example: PRIMA-162 2P+F gyv 16A fehér dugaj IP20 730,00 177,39 834,39
            # This has: product_name unit_price list_price gross
            elif re.search(r'\d+[,\.]\d+', line):  # Line has numbers (potential item)
                # Extract all numbers from the line
                numbers = re.findall(r'\d+[,\.]\d+', line)

                # Single-row format typically has 3 numbers: unit_price, list_price, gross
                # Or sometimes 2 numbers: net, gross
                if len(numbers) >= 2:
                    # Find where the numbers start
                    first_num = re.search(r'\d+[,\.]\d+', line)
                    if first_num:
                        product_name = line[:first_num.start()].strip()

                        # Filter out false positives:
                        # 1. Lines with no product name (just numbers)
                        # 2. Lines starting with lowercase (continuation lines)
                        # 3. Lines with very short product names (< 5 chars)
                        # 4. Lines containing only technical specs or units
                        # 5. Lines containing page markers or totals
                        if not product_name:
                            output_lines.append(line)
                        elif len(product_name) < 5:
                            output_lines.append(line)
                        elif product_name[0].islower():
                            output_lines.append(line)
                        elif re.match(r'^[\d\s\-/\.]+$', product_name):  # Only digits, spaces, dashes, slashes, dots
                            output_lines.append(line)
                        elif re.search(r'Oldal:|Osszesen:|Total:|Page:', product_name, re.IGNORECASE):
                            output_lines.append(line)
                        elif re.match(r'^\d+[-/]?\d+[A-Z]', product_name):  # Technical specs like "120-250V" or "100/200A"
                            output_lines.append(line)
                        else:
                            # Valid item line - add table header if not done yet
                            if not table_header_added:
                                output_lines.append("\nMegnevezés | Mennyiség | Egységár | Nettó | Bruttó")
                                table_header_added = True
                                items_section_started = True

                            # Convert numbers to proper decimal format
                            numbers = [n.replace(',', '.') for n in numbers]

                            # For single-row format:
                            # If 3+ numbers: [unit_price, list_price, gross]
                            # If 2 numbers: [net, gross]
                            if len(numbers) >= 3:
                                unit_price = numbers[0]
                                net = numbers[1]  # List price
                                gross = numbers[-1]  # Last number is always gross
                            elif len(numbers) == 2:
                                unit_price = ""
                                net = numbers[0]
                                gross = numbers[1]
                            else:
                                unit_price = ""
                                net = numbers[0]
                                gross = numbers[0]

                            # Format as table row (no quantity for single-row format)
                            table_row = f"{product_name} | | {unit_price} | {net} | {gross}"
                            output_lines.append(table_row)
                            logger.debug(f"Converted PATTERN 2 to table row: {table_row}")
                    else:
                        # Couldn't parse, keep as is
                        output_lines.append(line)
                else:
                    # Not enough numbers, keep as is
                    output_lines.append(line)
            else:
                # Not an item line, keep as is
                output_lines.append(line)

            i += 1

        return '\n'.join(output_lines)

    def _extract_with_table_structure(self, image: Image.Image) -> str:
        """
        Extract text with proper table structure using TSV output.
        Groups words into columns based on horizontal positioning.

        Args:
            image: PIL Image to extract text from

        Returns:
            Text with table structure preserved (tab-separated columns)
        """
        try:
            # Get TSV data with positioning information
            tsv_data = pytesseract.image_to_data(
                image,
                lang='hun+eng',
                config=r'--psm 6 --oem 3',
                output_type=pytesseract.Output.DICT
            )

            # Group words by their vertical position (row)
            rows = {}

            for i in range(len(tsv_data['text'])):
                text = tsv_data['text'][i].strip()
                if not text:
                    continue

                conf = int(tsv_data['conf'][i])
                if conf < 30:  # Skip low confidence
                    continue

                # Use top position to group into rows (with some tolerance)
                top = tsv_data['top'][i]
                left = tsv_data['left'][i]
                height = tsv_data['height'][i]

                # Find or create row (group by vertical position with tolerance)
                row_key = self._find_row_key(rows, top, height, tolerance=10)

                if row_key not in rows:
                    rows[row_key] = []

                rows[row_key].append({
                    'text': text,
                    'left': left,
                    'top': top
                })

            # Sort rows by vertical position
            sorted_rows = sorted(rows.items(), key=lambda x: x[0])

            # Detect column boundaries by analyzing word positions across all rows
            column_positions = self._detect_columns(rows)

            # Format rows with column alignment
            formatted_lines = []
            for row_top, words in sorted_rows:
                # Sort words by horizontal position
                sorted_words = sorted(words, key=lambda x: x['left'])

                # Assign words to columns
                row_text = self._format_row_with_columns(sorted_words, column_positions)
                formatted_lines.append(row_text)

            return '\n'.join(formatted_lines)

        except Exception as e:
            logger.error(f"Table structure extraction failed: {e}")
            # Fallback to simple text extraction
            return pytesseract.image_to_string(image, lang='hun+eng', config=r'--psm 6 --oem 3')

    def _find_row_key(self, rows: dict, top: int, height: int, tolerance: int = 10) -> int:
        """Find existing row key or return new one based on vertical position"""
        for existing_top in rows.keys():
            if abs(existing_top - top) <= tolerance:
                return existing_top
        return top

    def _detect_columns(self, rows: dict) -> list:
        """
        Detect column boundaries by analyzing horizontal positions of words.
        Returns list of column left positions.
        """
        # Collect all left positions with their frequency
        position_counts = {}
        for words in rows.values():
            for word in words:
                left = word['left']
                # Group positions within small range (±20 pixels)
                found_bucket = False
                for bucket_pos in position_counts.keys():
                    if abs(bucket_pos - left) <= 20:
                        position_counts[bucket_pos] += 1
                        found_bucket = True
                        break
                if not found_bucket:
                    position_counts[left] = 1

        if not position_counts:
            return [0]

        # Sort by position
        sorted_positions = sorted(position_counts.items(), key=lambda x: x[0])

        # Find clusters of positions (columns) - use larger gap for actual column separation
        columns = [sorted_positions[0][0]]
        min_column_gap = 100  # Increased from 50 - larger gap between actual columns

        for pos, count in sorted_positions:
            if pos - columns[-1] > min_column_gap:
                columns.append(pos)

        logger.info(f"Detected {len(columns)} columns at positions: {columns}")
        return columns

    def _format_row_with_columns(self, words: list, column_positions: list) -> str:
        """
        Format a row of words according to detected column positions.
        Uses tabs to separate columns.
        """
        # Assign each word to nearest column
        columns_content = {col: [] for col in column_positions}

        for word in words:
            # Find nearest column
            nearest_col = min(column_positions, key=lambda col: abs(col - word['left']))
            columns_content[nearest_col].append(word['text'])

        # Build row with tab-separated columns
        row_parts = []
        for col in column_positions:
            col_text = ' '.join(columns_content[col])
            if col_text:
                row_parts.append(col_text)

        return '\t'.join(row_parts) if row_parts else ''
