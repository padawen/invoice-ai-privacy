import json
import logging
import requests
import time
from typing import Dict, Any, Optional
from json_repair import repair_json
from config import Config

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for communicating with Ollama LLM"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.base_url = f"http://{self.config.OLLAMA_HOST}"
        self.model = self.config.OLLAMA_MODEL
        self.timeout = self.config.OLLAMA_TIMEOUT

    def health_check(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.error(f"Ollama not responding: {response.status_code}")
                return False

            # Check if model is available
            models = response.json().get('models', [])
            model_names = [model.get('name', '') for model in models]

            if not any(self.model.split(':')[0] in name for name in model_names):
                logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                # Try to pull the model
                return self.pull_model()

            return True

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def pull_model(self) -> bool:
        """Pull the required model if not available"""
        try:
            logger.info(f"Pulling model {self.model}...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300  # 5 minutes timeout for model download
            )

            if response.status_code == 200:
                logger.info(f"Successfully pulled model {self.model}")
                return True
            else:
                logger.error(f"Failed to pull model: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Model pull failed: {str(e)}")
            return False

    def generate_completion(self, prompt: str) -> Dict[str, Any]:
        """
        Generate completion using Ollama

        Args:
            prompt: Input prompt for the LLM

        Returns:
            Dict containing the structured invoice data
        """
        try:
            logger.info("Starting LLM processing...")

            # Prepare the request
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent structured output
                    "top_p": 0.9,
                    "num_predict": 2048,  # Max tokens for response
                }
            }

            # Make request to Ollama
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

            response_data = response.json()
            raw_response = response_data.get('response', '')

            processing_time = time.time() - start_time
            logger.info(f"LLM processing completed in {processing_time:.2f} seconds")

            if not raw_response:
                raise Exception("Empty response from Ollama")

            # Parse the JSON response
            structured_data = self.parse_llm_response(raw_response)
            return structured_data

        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            raise Exception(f"LLM processing timed out after {self.timeout} seconds")
        except Exception as e:
            logger.error(f"LLM processing failed: {str(e)}")
            raise Exception(f"LLM processing failed: {str(e)}")

    def parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON structure

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed invoice data dictionary
        """
        try:
            # Clean the response text
            cleaned_response = response_text.strip()

            # Remove code block markers if present
            if cleaned_response.startswith('```json') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[7:-3].strip()
            elif cleaned_response.startswith('```') and cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[3:-3].strip()

            # Extract JSON from text
            json_match = self.extract_json_from_text(cleaned_response)
            if not json_match:
                raise ValueError("No valid JSON found in LLM response")

            # Try to parse JSON
            try:
                parsed_data = json.loads(json_match)
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed, attempting repair: {str(e)}")
                # Try to repair malformed JSON
                repaired_json = repair_json(json_match)
                parsed_data = json.loads(repaired_json)

            # Validate structure
            validated_data = self.validate_and_normalize(parsed_data)
            return validated_data

        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            logger.error(f"Raw response: {response_text}")

            # Return fallback structure
            return self.get_fallback_structure()

    def extract_json_from_text(self, text: str) -> Optional[str]:
        """Extract JSON object from text"""
        import re

        # Find JSON-like structure
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            # Return the largest/most complete match
            return max(matches, key=len)

        return None

    def validate_and_normalize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize the parsed data to match expected structure

        Args:
            data: Raw parsed data

        Returns:
            Normalized data matching the expected structure
        """
        try:
            # Ensure required top-level structure
            normalized = {
                "seller": self.normalize_seller(data.get("seller", {})),
                "buyer": self.normalize_buyer(data.get("buyer", {})),
                "invoice_number": str(data.get("invoice_number", "")),
                "issue_date": str(data.get("issue_date", "")),
                "fulfillment_date": str(data.get("fulfillment_date", "")),
                "due_date": str(data.get("due_date", "")),
                "payment_method": str(data.get("payment_method", "")),
                "currency": str(data.get("currency", "")),
                "invoice_data": self.normalize_invoice_items(data.get("invoice_data", []))
            }

            return normalized

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return self.get_fallback_structure()

    def normalize_seller(self, seller_data: Any) -> Dict[str, str]:
        """Normalize seller data"""
        if not isinstance(seller_data, dict):
            seller_data = {}

        return {
            "name": str(seller_data.get("name", "")),
            "address": str(seller_data.get("address", "")),
            "tax_id": str(seller_data.get("tax_id", "")),
            "email": str(seller_data.get("email", "")),
            "phone": str(seller_data.get("phone", ""))
        }

    def normalize_buyer(self, buyer_data: Any) -> Dict[str, str]:
        """Normalize buyer data"""
        if not isinstance(buyer_data, dict):
            buyer_data = {}

        return {
            "name": str(buyer_data.get("name", "")),
            "address": str(buyer_data.get("address", "")),
            "tax_id": str(buyer_data.get("tax_id", ""))
        }

    def normalize_invoice_items(self, items_data: Any) -> list:
        """Normalize invoice items array"""
        if not isinstance(items_data, list):
            return []

        normalized_items = []
        for item in items_data:
            if isinstance(item, dict):
                normalized_item = {
                    "name": str(item.get("name", "")),
                    "quantity": str(item.get("quantity", "")),
                    "unit_price": str(item.get("unit_price", "")),
                    "net": str(item.get("net", "")),
                    "gross": str(item.get("gross", "")),
                    "currency": str(item.get("currency", ""))
                }
                normalized_items.append(normalized_item)

        return normalized_items

    def get_fallback_structure(self) -> Dict[str, Any]:
        """Return fallback structure when parsing fails"""
        return {
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

    def create_extraction_prompt(self, ocr_text: str) -> str:
        """
        Create prompt for invoice data extraction

        Args:
            ocr_text: Text extracted from PDF via OCR

        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""
Extract all relevant data from the following invoice text and return only the JSON result in the exact format shown below, in English.

CRITICAL INSTRUCTIONS:
1. You MUST return a valid JSON object in the exact format specified below, even if some fields are empty
2. If any field cannot be found, use an empty string "" for that field - NEVER omit the field entirely
3. The response must be ONLY the JSON object - no explanations, notes, or additional text

Data Extraction Instructions:
1. Identify all invoice line items from the text. Each product/service must be a separate entry.
2. For each line item, extract:
   - "name": product or service name
   - "quantity": quantity (numeric, as string)
   - "unit_price": unit price (numeric, as string)
   - "net": net value (numeric, as string)
   - "gross": gross value (numeric, as string)
   - "currency": currency code (use "HUF" if you see "Ft" or "ft")
3. For all numeric values, remove spaces and currency symbols, convert commas to periods
4. Extract seller, buyer information and invoice metadata

MANDATORY JSON format:
{{
  "seller": {{
    "name": "",
    "address": "",
    "tax_id": "",
    "email": "",
    "phone": ""
  }},
  "buyer": {{
    "name": "",
    "address": "",
    "tax_id": ""
  }},
  "invoice_number": "",
  "issue_date": "",
  "fulfillment_date": "",
  "due_date": "",
  "payment_method": "",
  "currency": "",
  "invoice_data": [
    {{
      "name": "",
      "quantity": "",
      "unit_price": "",
      "net": "",
      "gross": "",
      "currency": ""
    }}
  ]
}}

Invoice text to process:

{ocr_text}

Return ONLY the JSON object:"""

        return prompt