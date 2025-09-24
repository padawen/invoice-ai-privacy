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

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """
        Extract text from PDF using OCR

        Args:
            pdf_bytes: PDF file as bytes

        Returns:
            Extracted text as string
        """
        try:
            logger.info("Starting PDF to text extraction")

            # Convert PDF to images
            images = self.pdf_to_images(pdf_bytes)
            if not images:
                raise ValueError("Could not convert PDF to images")

            # Extract text from all images
            extracted_texts = []
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")

                # Preprocess image for better OCR
                processed_image = self.preprocess_image(image)

                # Extract text
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

            # Apply Gaussian blur to remove noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply threshold to get binary image
            _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)

            # Convert back to PIL
            processed_image = Image.fromarray(cleaned)

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

    def clean_text(self, text: str) -> str:
        """
        Clean extracted text

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines

        # Join lines back
        cleaned = '\n'.join(lines)

        return cleaned

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