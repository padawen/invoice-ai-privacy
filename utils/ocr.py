import os
import tempfile
import logging
from typing import Optional, List
from PIL import Image
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from config import Config

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles PDF to text conversion using docTR OCR"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.predictor = None
        self.setup_doctr()

    def setup_doctr(self):
        """Initialize docTR model"""
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor

            # Initialize the OCR predictor with multilingual support
            # docTR supports Hungarian and other European languages
            logger.debug("Initializing docTR OCR model with multilingual support...")

            # Use the recognition model with multilingual support
            # The 'latin' vocab includes Hungarian characters
            self.predictor = ocr_predictor(
                det_arch='db_resnet50',  # Detection architecture
                reco_arch='crnn_vgg16_bn',  # Recognition architecture
                pretrained=True,
                assume_straight_pages=True,
                preserve_aspect_ratio=True,
                symmetric_pad=True
            )
            logger.debug("docTR model initialized successfully with multilingual support")

        except Exception as e:
            logger.error(f"Failed to initialize docTR: {str(e)}")
            raise Exception(f"docTR initialization failed: {str(e)}")

    def extract_text_from_pdf(self, pdf_bytes: bytes, job_id: str = None, structured: bool = False) -> str:
        """
        Extract text from PDF using docTR OCR

        Args:
            pdf_bytes: PDF file as bytes
            job_id: Optional job ID for cancellation checking
            structured: If True, preserve document structure (for line items)

        Returns:
            Extracted text as string
        """
        try:
            logger.info(f"Starting PDF to text extraction with docTR (structured={structured})")

            # Check for cancellation before starting
            if job_id:
                from .progress import progress_tracker
                if progress_tracker.is_cancelled(job_id):
                    logger.info(f"Job {job_id} was cancelled before OCR")
                    raise Exception("Processing was cancelled by user")

            # Convert PDF to images
            images = self.pdf_to_images(pdf_bytes)
            if not images:
                raise ValueError("Could not convert PDF to images")

            # Extract text from all images using docTR
            extracted_texts = []
            for i, image in enumerate(images):
                # Check for cancellation before processing each page
                if job_id:
                    from .progress import progress_tracker
                    if progress_tracker.is_cancelled(job_id):
                        logger.info(f"Job {job_id} was cancelled during OCR processing (page {i+1})")
                        raise Exception("Processing was cancelled by user")

                logger.info(f"Processing page {i+1}/{len(images)} with docTR")

                # Preprocess image for better OCR
                processed_image = self.preprocess_image(image)

                # Extract text using docTR
                text = self.image_to_text(processed_image, structured=structured)

                if text.strip():
                    extracted_texts.append(text)

            # Combine all text
            full_text = "\n\n".join(extracted_texts)
            logger.info(f"Extracted {len(full_text)} characters from {len(images)} pages")

            return full_text

        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise Exception(f"OCR processing failed: {str(e)}")

    def pdf_to_images(self, pdf_bytes: bytes) -> List[Image.Image]:
        """Convert PDF bytes to PIL Images"""
        try:
            images = convert_from_bytes(
                pdf_bytes,
                dpi=self.config.OCR_DPI,
                fmt='png'
            )
            logger.info(f"Converted PDF to {len(images)} images")
            return images

        except Exception as e:
            logger.error(f"PDF to image conversion failed: {str(e)}")
            raise Exception(f"PDF to image conversion failed: {str(e)}")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy

        Args:
            image: PIL Image

        Returns:
            Preprocessed PIL Image
        """
        try:
            # For docTR, simpler preprocessing works better
            # Just ensure RGB mode without aggressive thresholding
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Optional: slight upscaling for better character recognition
            # Increase size by 50% for better detail
            width, height = image.size
            new_size = (int(width * 1.5), int(height * 1.5))
            image = image.resize(new_size, Image.LANCZOS)

            return image

        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            # Ensure RGB mode even for original image
            if image.mode != 'RGB':
                return image.convert('RGB')
            return image

    def image_to_text(self, image: Image.Image, structured: bool = False) -> str:
        """
        Extract text from a single image using docTR

        Args:
            image: PIL Image
            structured: If True, preserve spatial structure

        Returns:
            Extracted text
        """
        try:
            # Convert PIL Image to numpy array for docTR
            image_array = np.array(image)

            # Run OCR prediction
            result = self.predictor([image_array])

            # Extract text based on structure preference
            if structured:
                text = self.extract_structured_text(result)
            else:
                text = self.extract_plain_text(result)

            # Clean up text
            cleaned_text = self.clean_text(text)

            return cleaned_text

        except Exception as e:
            logger.error(f"Text extraction from image failed: {str(e)}")
            return ""

    def fix_ocr_characters(self, text: str) -> str:
        """
        Fix common OCR character recognition errors for Hungarian text

        docTR recognizes all Hungarian accented characters as 'é' (U+00E9)
        This function replaces them with non-accented equivalents

        Args:
            text: Raw OCR text

        Returns:
            Corrected text
        """
        import re

        # docTR replaces all accented chars (á, é, í, ó, ö, ő, ú, ü, ű) with 'é'
        # Strategy: Replace 'é' with 'e' in most cases for simplicity

        # First, handle specific word patterns where we know the correct spelling
        word_fixes = {
            # Common product name patterns
            'Betét': 'Betet',
            'betét': 'betet',
            'Tusfürdő': 'Tusfurdo',
            'tusfürdő': 'tusfurdo',
            'Óvszer': 'Ovszer',
            'óvszer': 'ovszer',
            'Babakrém': 'Babakrem',
            'babakrém': 'babakrem',
            'Folyékony': 'Folyekony',
            'folyékony': 'folyekony',
            'Fogkrém': 'Fogkrem',
            'fogkrém': 'fogkrem',
            'Szájvíz': 'Szajviz',
            'szájvíz': 'szajviz',
        }

        # Apply word-level fixes
        for wrong, correct in word_fixes.items():
            text = text.replace(wrong, correct)

        # Generic replacements for remaining accented characters
        # Replace all 'é' with 'e' (since docTR uses 'é' for all accents)
        text = text.replace('é', 'e')
        text = text.replace('É', 'E')

        # Also handle other potential accented characters
        accented_to_plain = {
            'á': 'a', 'Á': 'A',
            'í': 'i', 'Í': 'I',
            'ó': 'o', 'Ó': 'O',
            'ö': 'o', 'Ö': 'O',
            'ő': 'o', 'Ő': 'O',
            'ú': 'u', 'Ú': 'U',
            'ü': 'u', 'Ü': 'U',
            'ű': 'u', 'Ű': 'U',
        }

        for accented, plain in accented_to_plain.items():
            text = text.replace(accented, plain)

        return text

    def extract_plain_text(self, result) -> str:
        """
        Extract plain text from docTR result

        Args:
            result: docTR OCR result object

        Returns:
            Plain text string
        """
        try:
            # Extract text from the result
            text_lines = []

            # Navigate through the docTR result structure
            # result.pages[0] -> page
            # page.blocks -> text blocks
            # block.lines -> text lines
            # line.words -> words

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join([word.value for word in line.words])
                        if line_text.strip():
                            text_lines.append(line_text)

            text = "\n".join(text_lines)

            # Fix common OCR errors
            text = self.fix_ocr_characters(text)

            return text

        except Exception as e:
            logger.error(f"Plain text extraction failed: {str(e)}")
            return ""

    def detect_column_layout(self, result) -> dict:
        """
        Detect if the document has a two-column layout using X-coordinate clustering.

        Args:
            result: docTR OCR result object

        Returns:
            dict with keys:
                - 'has_columns': bool
                - 'threshold': float (X coordinate dividing columns, e.g., 0.45)
                - 'left_words': list of word dicts
                - 'right_words': list of word dicts
        """
        try:
            # Collect all words with their X positions
            all_words = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_y = line.geometry[0][1]  # Y position of line
                        for word in line.words:
                            word_x = word.geometry[0][0]  # X position of word
                            all_words.append({
                                'value': word.value,
                                'x': word_x,
                                'y': line_y,
                                'confidence': word.confidence
                            })

            if len(all_words) < 10:
                return {'has_columns': False}

            # Extract X positions
            x_positions = [w['x'] for w in all_words]

            # Simple heuristic: Check if there's a clear gap in X positions
            # Sort X positions and look for the largest gap in the middle 50%
            sorted_x = sorted(x_positions)
            mid_start = len(sorted_x) // 4
            mid_end = 3 * len(sorted_x) // 4

            max_gap = 0
            gap_position = 0.5

            for i in range(mid_start, mid_end):
                gap = sorted_x[i + 1] - sorted_x[i]
                if gap > max_gap:
                    max_gap = gap
                    gap_position = (sorted_x[i] + sorted_x[i + 1]) / 2

            # If the gap is significant (> 0.15 of page width), we have columns
            has_columns = max_gap > 0.15

            if has_columns:
                # Separate words into left and right columns
                left_words = [w for w in all_words if w['x'] < gap_position]
                right_words = [w for w in all_words if w['x'] >= gap_position]

                logger.info(f"Detected two-column layout with threshold X={gap_position:.2f}")
                logger.info(f"Left column: {len(left_words)} words, Right column: {len(right_words)} words")

                return {
                    'has_columns': True,
                    'threshold': gap_position,
                    'left_words': left_words,
                    'right_words': right_words
                }
            else:
                return {'has_columns': False}

        except Exception as e:
            logger.error(f"Column detection failed: {str(e)}")
            return {'has_columns': False}

    def extract_structured_text(self, result) -> str:
        """
        Extract structured text from docTR result, preserving spatial layout.

        NOTE: Two-column detection disabled - it was causing issues with
        seller/buyer data extraction and line items.

        Args:
            result: docTR OCR result object

        Returns:
            Structured text string
        """
        try:
            # Always use single-column extraction (two-column detection disabled)
            return self._extract_single_column_text(result)

        except Exception as e:
            logger.error(f"Structured text extraction failed: {str(e)}")
            # Fallback to plain text
            return self.extract_plain_text(result)

    def _extract_single_column_text(self, result) -> str:
        """
        Extract text from single-column layout with better table structure preservation

        This method now uses word-level extraction instead of line-level to better
        handle table layouts where columns need proper spacing preservation.
        """
        try:
            text_lines = []

            # Extract text with spatial information at word level
            for page in result.pages:
                # Collect all words with their positions
                all_words = []

                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            word_geometry = word.geometry
                            all_words.append({
                                'value': word.value,
                                'x': word_geometry[0][0],  # x-coordinate of top-left
                                'y': word_geometry[0][1],  # y-coordinate of top-left
                                'confidence': word.confidence
                            })

                # Group words into lines based on Y-coordinate proximity
                # Use a tighter threshold to avoid merging separate rows
                y_threshold = 0.008  # Words within 0.8% of page height are on same line

                # Sort words by Y first, then X
                all_words.sort(key=lambda w: (w['y'], w['x']))

                # Group words into lines
                page_lines = []
                current_line_words = []
                current_y = None

                for word in all_words:
                    if current_y is None or abs(word['y'] - current_y) < y_threshold:
                        # Same line
                        current_line_words.append(word)
                        if current_y is None:
                            current_y = word['y']
                    else:
                        # New line - save previous line
                        if current_line_words:
                            # Sort words in line by X position (left to right)
                            current_line_words.sort(key=lambda w: w['x'])

                            # Build line text with spacing preservation
                            line_text = self._build_line_with_spacing(current_line_words)

                            if line_text.strip():
                                page_lines.append({
                                    'text': line_text,
                                    'y': current_y
                                })

                        # Start new line
                        current_line_words = [word]
                        current_y = word['y']

                # Add last line
                if current_line_words:
                    current_line_words.sort(key=lambda w: w['x'])
                    line_text = self._build_line_with_spacing(current_line_words)
                    if line_text.strip():
                        page_lines.append({
                            'text': line_text,
                            'y': current_y
                        })

                # Sort lines by Y position (top to bottom)
                page_lines.sort(key=lambda l: l['y'])
                text_lines.extend([l['text'] for l in page_lines])

            text = "\n".join(text_lines)

            # Fix common OCR errors
            text = self.fix_ocr_characters(text)

            return text

        except Exception as e:
            logger.error(f"Single column text extraction failed: {str(e)}")
            return ""

    def _build_line_with_spacing(self, words: List[dict]) -> str:
        """
        Build line text with proper spacing between words based on X-coordinates

        This helps preserve table column structure by adding extra spaces
        between words that are far apart horizontally.

        Args:
            words: List of word dicts with 'value' and 'x' keys, sorted by X

        Returns:
            Line text with proper spacing
        """
        if not words:
            return ""

        if len(words) == 1:
            return words[0]['value']

        # Build line with spacing
        line_parts = [words[0]['value']]

        for i in range(1, len(words)):
            prev_x = words[i-1]['x']
            curr_x = words[i]['x']

            # Calculate horizontal gap
            gap = curr_x - prev_x

            # If gap is large (> 3% of page width), add extra space
            # This helps preserve table column alignment
            if gap > 0.03:
                line_parts.append("  " + words[i]['value'])  # Double space
            else:
                line_parts.append(words[i]['value'])  # Single space

        return " ".join(line_parts)


    def _extract_two_column_text(self, column_info: dict) -> str:
        """
        Extract text from two-column layout, keeping columns separate.

        Args:
            column_info: Dict with 'left_words' and 'right_words' lists

        Returns:
            Text with columns properly separated
        """
        try:
            def build_text_from_words(words):
                """Group words into lines based on Y position, then sort by Y"""
                if not words:
                    return ""

                # Sort words by Y position first
                sorted_words = sorted(words, key=lambda w: (w['y'], w['x']))

                # Group words into lines (words with similar Y are on same line)
                lines = []
                current_line = []
                current_y = sorted_words[0]['y']
                y_threshold = 0.015  # Words within 1.5% of page height are on same line

                for word in sorted_words:
                    if abs(word['y'] - current_y) < y_threshold:
                        # Same line
                        current_line.append(word)
                    else:
                        # New line
                        if current_line:
                            # Sort words in line by X position
                            current_line.sort(key=lambda w: w['x'])
                            line_text = " ".join([w['value'] for w in current_line])
                            lines.append(line_text)
                        current_line = [word]
                        current_y = word['y']

                # Add last line
                if current_line:
                    current_line.sort(key=lambda w: w['x'])
                    line_text = " ".join([w['value'] for w in current_line])
                    lines.append(line_text)

                return "\n".join(lines)

            # Extract text from each column
            left_text = build_text_from_words(column_info['left_words'])
            right_text = build_text_from_words(column_info['right_words'])

            # Combine columns with clear separation
            # Use a marker that LLM can understand
            combined_text = f"=== LEFT COLUMN ===\n{left_text}\n\n=== RIGHT COLUMN ===\n{right_text}"

            # Fix common OCR errors
            combined_text = self.fix_ocr_characters(combined_text)

            logger.info(f"Two-column text extracted: {len(left_text)} chars (left), {len(right_text)} chars (right)")

            return combined_text

        except Exception as e:
            logger.error(f"Two-column text extraction failed: {str(e)}")
            return ""

    def is_barcode(self, number_str: str) -> bool:
        """
        Detect if a number string is likely a barcode (EAN-8, EAN-13, UPC, etc.)

        Args:
            number_str: Number string to check

        Returns:
            True if likely a barcode, False otherwise
        """
        import re
        # Remove any non-digit characters
        digits_only = re.sub(r'\D', '', number_str)

        # Common barcode lengths: 8 (EAN-8), 12 (UPC), 13 (EAN-13), 14 (ITF-14)
        barcode_lengths = [8, 12, 13, 14]

        return len(digits_only) in barcode_lengths and digits_only.isdigit()

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text and extract relevant invoice sections

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned and filtered text focusing on invoice data
        """
        if not text:
            return ""

        # Fix Hungarian number format issues FIRST (before splitting lines)
        import re
        # Only fix spaces in price-like patterns (1 244,00 → 1244,00)
        # Match: digit + space + 3 digits + comma/period + 2 digits
        text = re.sub(r'(\d)[\s\u00A0\u202F\u2009]+(\d{3}[,.])', r'\1\2', text)
        # Also handle thousand separators without decimals (1 244 → 1244) but only for 3-4 digit groups
        text = re.sub(r'(\d)[\s\u00A0\u202F\u2009]+(\d{3})(?=\D|$)', r'\1\2', text)

        # Fix concatenated numbers (e.g., "1236,00236,00" -> "1236,00 236,00")
        # Pattern: number with 2 decimal places followed immediately by another digit
        text = re.sub(r'(\d+[,.]\d{2})(\d)', r'\1 \2', text)

        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines

        # Smart filtering: Remove common noise patterns
        filtered_lines = []
        skip_patterns = [
            'terms and conditions',
            'privacy policy',
            'general terms',
            'disclaimer',
            'footer',
            'page \d+ of \d+',
            'copyright',
            'all rights reserved',
            'www\.',
            'https?://',
        ]

        import re
        for line in lines:
            line_lower = line.lower()
            # Skip lines matching noise patterns
            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, line_lower):
                    should_skip = True
                    break

            # Keep lines with important invoice data markers
            has_important_data = any([
                re.search(r'\d{4}[-./]\d{1,2}[-./]\d{1,2}', line),  # Dates
                re.search(r'invoice|receipt|bill|számlaszám', line_lower),  # Invoice markers
                re.search(r'\d+[.,]\d+', line),  # Numbers (prices)
                re.search(r'tax|vat|áfa|nettó|bruttó', line_lower),  # Tax info and totals
                re.search(r'total|subtotal|összesen|mennyiség', line_lower),  # Totals and quantity
                re.search(r'[A-Z]{2,}[-\s]?\d{8,}', line),  # Tax IDs
                re.search(r'egységár|unit', line_lower),  # Unit price headers
            ])

            if not should_skip or has_important_data:
                filtered_lines.append(line)

        # Join lines back
        cleaned = '\n'.join(filtered_lines)

        return cleaned

    def extract_table_structure(self, image: Image.Image) -> str:
        """
        Extract table rows with row continuation capture for multi-line names
        This uses the structured text extraction to better preserve table layout

        Args:
            image: PIL Image

        Returns:
            Filtered text with table data rows (including continuation lines)
        """
        try:
            import re

            # Get structured text (preserves spatial layout)
            full_text = self.image_to_text(image, structured=True)
            lines = full_text.split('\n')

            # Filter to lines that look like table rows + capture continuations
            table_lines = []
            skip_next = 0  # Track lines to skip (already appended as continuations)

            for i, line in enumerate(lines):
                # Skip if this line was already appended as a continuation
                if skip_next > 0:
                    skip_next -= 1
                    continue

                # Count price-like patterns in the line
                price_count = len(re.findall(r'\d+[.,]\d+', line))

                # Skip summary/total rows (mostly numbers, no real text)
                words = re.findall(r'[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]{3,}', line)

                # Check if line is a header/summary (skip from continuation capture)
                is_header_or_summary = re.search(
                    r'^\s*(total|nettó|bruttó|áfa|összesen|sum|subtotal)\s*:?\s*\d',
                    line,
                    re.IGNORECASE
                )

                # Skip if only numbers (summary row)
                if price_count >= 3 and len(words) == 0:
                    continue

                # If line has 3+ prices and some text, it's likely a data row
                if price_count >= 3 and len(words) > 0:
                    combined_line = line

                    # Capture next 1-2 lines as name continuation if they're text-only
                    continuation_count = 0
                    for j in range(1, 3):  # Check next 2 lines
                        if i + j >= len(lines):
                            break

                        next_line = lines[i + j].strip()
                        if not next_line:
                            continue

                        # Check if next line is text-only (no prices, not header/summary)
                        next_prices = len(re.findall(r'\d+[.,]\d+', next_line))
                        next_words = re.findall(r'[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]{3,}', next_line)
                        is_next_header = re.search(
                            r'(total|nettó|bruttó|áfa|összesen|sum|subtotal|pozíció|leírás)',
                            next_line,
                            re.IGNORECASE
                        )

                        # Append if: has text, no prices, not header/summary
                        if next_prices == 0 and len(next_words) > 0 and not is_next_header:
                            combined_line += " " + next_line
                            continuation_count += 1
                        else:
                            break  # Stop at first non-text line

                    table_lines.append(combined_line)
                    skip_next = continuation_count

                # Also keep header row if it has quantity/price keywords
                elif re.search(r'pozíció|leírás|mennyiség|egységár|nettó|bruttó|quantity|unit|net|gross', line, re.IGNORECASE):
                    table_lines.append(line)

            filtered_text = '\n'.join(table_lines)
            logger.info(f"Filtered to {len(table_lines)} table rows from {len(lines)} total lines")

            # If we got too few table lines (< 2), fallback to full text
            if len(table_lines) < 2:
                logger.warning(f"Too few table rows ({len(table_lines)}); falling back to full text for items extraction")
                return full_text

            return filtered_text

        except Exception as e:
            logger.warning(f"Table filtering failed, using plain text: {str(e)}")
            return self.image_to_text(image, structured=False)

    def get_text_confidence(self, image: Image.Image) -> float:
        """
        Get OCR confidence score for an image using docTR

        Args:
            image: PIL Image

        Returns:
            Confidence score (0-100)
        """
        try:
            # Convert PIL Image to numpy array
            image_array = np.array(image)

            # Run OCR prediction
            result = self.predictor([image_array])

            # Calculate average confidence from all words
            confidences = []
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            confidences.append(word.confidence)

            if confidences:
                # docTR returns confidence as 0-1, convert to 0-100
                avg_confidence = sum(confidences) / len(confidences) * 100
                return avg_confidence
            return 0.0

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.0
