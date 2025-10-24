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
