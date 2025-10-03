import os
import tempfile
import logging
from typing import Optional, List
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import cv2
import numpy as np
from config import Config

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles PDF to text conversion using Pytesseract OCR"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.setup_tesseract()

    def setup_tesseract(self):
        """Configure Tesseract OCR settings"""
        # For different OS configurations
        if os.name == 'nt':  # Windows
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

        logger.info(f"Tesseract configured with language: {self.config.OCR_LANGUAGE}")

    def extract_text_from_pdf(self, pdf_bytes: bytes, job_id: str = None, structured: bool = False) -> str:
        """
        Extract text from PDF using OCR

        Args:
            pdf_bytes: PDF file as bytes
            job_id: Optional job ID for cancellation checking
            structured: If True, use table structure extraction (for line items)

        Returns:
            Extracted text as string
        """
        try:
            logger.info(f"Starting PDF to text extraction (structured={structured})")

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

            # Extract text from all images
            extracted_texts = []
            for i, image in enumerate(images):
                # Check for cancellation before processing each page
                if job_id:
                    from .progress import progress_tracker
                    if progress_tracker.is_cancelled(job_id):
                        logger.info(f"Job {job_id} was cancelled during OCR processing (page {i+1})")
                        raise Exception("Processing was cancelled by user")

                logger.info(f"Processing page {i+1}/{len(images)}")

                # Preprocess image for better OCR
                processed_image = self.preprocess_image(image)

                # Extract text (structured or plain)
                if structured:
                    text = self.extract_table_structure(processed_image)
                else:
                    text = self.image_to_text(processed_image)

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
            # Convert PIL to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

            # Apply simple threshold (faster and more reliable than complex preprocessing)
            _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Convert back to PIL
            processed_image = Image.fromarray(threshold)

            return processed_image

        except Exception as e:
            logger.warning(f"Image preprocessing failed, using original: {str(e)}")
            return image

    def image_to_text(self, image: Image.Image) -> str:
        """
        Extract text from a single image using Tesseract

        Args:
            image: PIL Image

        Returns:
            Extracted text
        """
        try:
            # Configure Tesseract
            custom_config = f'-l {self.config.OCR_LANGUAGE} {self.config.OCR_CONFIG}'

            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)

            # Clean up text
            cleaned_text = self.clean_text(text)

            return cleaned_text

        except Exception as e:
            logger.error(f"Text extraction from image failed: {str(e)}")
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

        Args:
            image: PIL Image

        Returns:
            Filtered text with table data rows (including continuation lines)
        """
        try:
            import re

            # Get full text first (plain OCR)
            full_text = self.image_to_text(image)
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
            return self.image_to_text(image)

    def get_text_confidence(self, image: Image.Image) -> float:
        """
        Get OCR confidence score for an image

        Args:
            image: PIL Image

        Returns:
            Confidence score (0-100)
        """
        try:
            custom_config = f'-l {self.config.OCR_LANGUAGE} {self.config.OCR_CONFIG}'
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)

            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            if confidences:
                return sum(confidences) / len(confidences)
            return 0.0

        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.0