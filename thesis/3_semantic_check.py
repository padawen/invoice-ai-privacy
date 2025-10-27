"""
Semantic Accuracy Testing Script

This script tests actual semantic correctness by comparing extracted data
against ground truth values, not just checking if fields are populated.

Usage:
    python thesis/semantic_accuracy_eval.py
"""
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.processing import InvoiceProcessor
from config import Config

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "thesis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DEFAULT = PROJECT_ROOT / "thesis_output" / "ground_truth_vision.json"
INVOICE_DIR = BASE_DIR / "invoice_templates"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress library logging
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)


class SemanticAccuracyTester:
    """Semantic accuracy testing with ground truth validation"""

    def __init__(self, invoice_dir: str | Path, ground_truth_path: str | Path):
        self.invoice_dir = Path(invoice_dir)
        self.ground_truth_path = Path(ground_truth_path)
        self.config = Config()
        self.processor = InvoiceProcessor(self.config)
        self.results = []

        # Load ground truth data
        with self.ground_truth_path.open('r', encoding='utf-8') as f:
            self.ground_truth = json.load(f)

    def run_semantic_tests(self):
        """Run semantic accuracy tests on all invoices with ground truth"""
        logger.info("=" * 80)
        logger.info("SEMANTIC ACCURACY TEST")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  Model: {self.config.OLLAMA_MODEL}")
        logger.info(f"  GPU Layers: {self.config.OLLAMA_NUM_GPU}")
        logger.info(f"  OCR DPI: {self.config.OCR_DPI}")
        logger.info(f"  Ground Truth: {len(self.ground_truth)} invoices")
        logger.info("=" * 80)
        logger.info("")

        # Process each invoice with ground truth
        for i, (filename, ground_truth) in enumerate(self.ground_truth.items(), 1):
            invoice_path = self.invoice_dir / filename

            if not invoice_path.exists():
                logger.warning(f"⚠️  Invoice not found: {filename}")
                continue

            logger.info(f"[{i}/{len(self.ground_truth)}] Testing: {filename}")
            result = self._test_invoice_semantic(invoice_path, ground_truth, filename)
            self.results.append(result)

            logger.info(f"  Status: {result['status']}")
            logger.info(f"  Semantic Accuracy: {result['semantic_accuracy']:.1f}%")
            logger.info(f"    - Seller: {result['seller_accuracy']:.1f}%")
            logger.info(f"    - Buyer: {result['buyer_accuracy']:.1f}%")
            logger.info(f"    - Metadata: {result['metadata_accuracy']:.1f}%")
            logger.info(f"    - Items: {result['items_accuracy']:.1f}%")
            logger.info("")

        # Generate summary
        self._generate_summary()

        # Save results
        self._save_results()

    def _test_invoice_semantic(self, invoice_path: str, ground_truth: Dict, filename: str) -> Dict:
        """Test a single invoice for semantic accuracy"""
        start_time = time.time()

        result = {
            'filename': filename,
            'filepath': str(invoice_path),
            'status': 'unknown',
            'total_duration': 0.0,
            'ocr_duration': 0.0,
            'llm_duration': 0.0,
            'semantic_accuracy': 0.0,
            'seller_accuracy': 0.0,
            'buyer_accuracy': 0.0,
            'metadata_accuracy': 0.0,
            'items_accuracy': 0.0,
            'errors': [],
            'field_errors': []
        }

        try:
            # Read and process invoice
            with invoice_path.open('rb') as f:
                pdf_bytes = f.read()

            extracted_data = self.processor.process_pdf(pdf_bytes, filename)

            # Extract timing
            metadata = extracted_data.get('_processing_metadata', {})
            result['total_duration'] = metadata.get('total_duration', time.time() - start_time)
            result['ocr_duration'] = metadata.get('ocr_duration', 0.0)
            result['llm_duration'] = metadata.get('llm_duration', 0.0)

            # Calculate semantic accuracy
            accuracy = self._calculate_semantic_accuracy(extracted_data, ground_truth)
            result['semantic_accuracy'] = accuracy['overall']
            result['seller_accuracy'] = accuracy['seller']
            result['buyer_accuracy'] = accuracy['buyer']
            result['metadata_accuracy'] = accuracy['metadata']
            result['items_accuracy'] = accuracy['items']
            result['field_errors'] = accuracy['field_errors']

            # Mark as error if below 95%
            if accuracy['overall'] < 95.0:
                result['errors'].append(f"Semantic accuracy below 95% threshold: {accuracy['overall']:.1f}%")

            result['status'] = 'success'

        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(str(e))
            result['total_duration'] = time.time() - start_time
            logger.error(f"  Error: {str(e)}")

        return result

    def _calculate_semantic_accuracy(self, extracted: Dict, ground_truth: Dict) -> Dict:
        """
        Calculate semantic accuracy by comparing extracted vs ground truth values

        Returns accuracy breakdown with field-level error details
        """
        accuracy = {
            'overall': 0.0,
            'seller': 0.0,
            'buyer': 0.0,
            'metadata': 0.0,
            'items': 0.0,
            'field_errors': []
        }

        # Seller accuracy
        seller_result = self._compare_entity(
            extracted.get('seller', {}),
            ground_truth.get('seller', {}),
            'seller'
        )
        accuracy['seller'] = seller_result['accuracy']
        accuracy['field_errors'].extend(seller_result['errors'])

        # Buyer accuracy
        buyer_result = self._compare_entity(
            extracted.get('buyer', {}),
            ground_truth.get('buyer', {}),
            'buyer'
        )
        accuracy['buyer'] = buyer_result['accuracy']
        accuracy['field_errors'].extend(buyer_result['errors'])

        # Metadata accuracy
        metadata_result = self._compare_metadata(extracted, ground_truth)
        accuracy['metadata'] = metadata_result['accuracy']
        accuracy['field_errors'].extend(metadata_result['errors'])

        # Items accuracy
        items_result = self._compare_items(
            extracted.get('invoice_data', []),
            ground_truth.get('invoice_data', []),
            extracted.get('currency', ''),
            ground_truth.get('currency', '')
        )
        accuracy['items'] = items_result['accuracy']
        accuracy['field_errors'].extend(items_result['errors'])

        # Overall accuracy (weighted average)
        accuracy['overall'] = (
            accuracy['seller'] * 0.25 +
            accuracy['buyer'] * 0.25 +
            accuracy['metadata'] * 0.25 +
            accuracy['items'] * 0.25
        )

        return accuracy

    def _compare_entity(self, extracted: Dict, expected: Dict, entity_name: str) -> Dict:
        """Compare seller or buyer data"""
        if not expected:
            return {'accuracy': 100.0, 'errors': []}

        fields = ['name', 'address', 'tax_id', 'email', 'phone']
        correct = 0
        total = 0
        errors = []

        for field in fields:
            expected_value = expected.get(field, '')
            extracted_value = extracted.get(field, '')

            # Skip empty expected values (optional fields)
            if not expected_value or expected_value.strip() == '':
                continue

            total += 1

            if self._values_match(extracted_value, expected_value, field=field):
                correct += 1
            else:
                errors.append({
                    'entity': entity_name,
                    'field': field,
                    'expected': expected_value,
                    'extracted': extracted_value,
                    'match': False
                })

        accuracy = (correct / total * 100) if total > 0 else 100.0
        return {'accuracy': accuracy, 'errors': errors}

    def _compare_metadata(self, extracted: Dict, expected: Dict) -> Dict:
        """Compare metadata fields"""
        fields = ['invoice_number', 'issue_date', 'fulfillment_date', 'due_date', 'payment_method', 'currency']
        correct = 0
        total = 0
        errors = []

        for field in fields:
            expected_value = expected.get(field, '')
            extracted_value = extracted.get(field, '')

            # Skip empty expected values (optional fields)
            if not expected_value or expected_value.strip() == '':
                continue

            total += 1

            if self._values_match(extracted_value, expected_value, field=field):
                correct += 1
            else:
                errors.append({
                    'entity': 'metadata',
                    'field': field,
                    'expected': expected_value,
                    'extracted': extracted_value,
                    'match': False
                })

        accuracy = (correct / total * 100) if total > 0 else 100.0
        return {'accuracy': accuracy, 'errors': errors}

    def _compare_items(
        self,
        extracted: List[Dict],
        expected: List[Dict],
        extracted_currency: str = "",
        expected_currency: str = ""
    ) -> Dict:
        """Compare line items"""
        if not expected:
            # No ground truth items defined
            return {'accuracy': 100.0, 'errors': []}

        if not extracted:
            return {
                'accuracy': 0.0,
                'errors': [{'entity': 'items', 'field': 'count', 'expected': len(expected), 'extracted': 0, 'match': False}]
            }

        # Match items by name similarity
        matched_pairs = self._match_items(extracted, expected)

        item_accuracies = []
        errors = []

        for idx, (extracted_item, expected_item) in enumerate(matched_pairs):
            if expected_item is None:
                # Extra item in extracted (not in ground truth)
                continue

            if extracted_item is None:
                # Missing item (in ground truth but not extracted)
                errors.append({
                    'entity': 'items',
                    'field': f'item_{idx+1}',
                    'expected': expected_item.get('name', ''),
                    'extracted': 'MISSING',
                    'match': False
                })
                item_accuracies.append(0.0)
                continue

            # Compare item fields
            item_result = self._compare_item(
                extracted_item,
                expected_item,
                idx + 1,
                expected_currency,
                extracted_currency
            )
            item_accuracies.append(item_result['accuracy'])
            errors.extend(item_result['errors'])

        overall_accuracy = sum(item_accuracies) / len(item_accuracies) if item_accuracies else 0.0
        return {'accuracy': overall_accuracy, 'errors': errors}

    def _match_items(self, extracted: List[Dict], expected: List[Dict]) -> List[Tuple]:
        """
        Match extracted items to expected items by name similarity
        Returns list of (extracted_item, expected_item) tuples
        """
        from difflib import SequenceMatcher

        matched_pairs = []
        used_extracted = set()
        used_expected = set()

        # First pass: exact or near-exact name matches
        for i, exp_item in enumerate(expected):
            exp_name = self._normalize_text(exp_item.get('name', ''))
            best_match = None
            best_score = 0.0

            for j, ext_item in enumerate(extracted):
                if j in used_extracted:
                    continue

                ext_name = self._normalize_text(ext_item.get('name', ''))
                score = SequenceMatcher(None, exp_name, ext_name).ratio()

                if score > best_score and score > 0.6:  # 60% similarity threshold
                    best_score = score
                    best_match = j

            if best_match is not None:
                matched_pairs.append((extracted[best_match], exp_item))
                used_extracted.add(best_match)
                used_expected.add(i)
            else:
                matched_pairs.append((None, exp_item))

        return matched_pairs

    def _compare_item(
        self,
        extracted: Dict,
        expected: Dict,
        item_num: int,
        expected_invoice_currency: str = "",
        extracted_invoice_currency: str = ""
    ) -> Dict:
        """Compare a single line item"""
        fields = ['quantity', 'unit_price', 'net', 'gross', 'currency']
        correct = 0
        total = 0
        errors = []

        for field in fields:
            expected_value = expected.get(field, '')
            extracted_value = extracted.get(field, '')

            # Skip empty expected values
            if not expected_value or str(expected_value).strip() == '':
                continue

            total += 1

            # For numeric fields, use fuzzy matching (±1% tolerance)
            if field in ['quantity', 'unit_price', 'net', 'gross']:
                if self._numbers_match(extracted_value, expected_value):
                    correct += 1
                else:
                    errors.append({
                        'entity': 'items',
                        'field': f'item_{item_num}.{field}',
                        'expected': expected_value,
                        'extracted': extracted_value,
                        'match': False
                    })
            else:
                if field == 'currency':
                    if self._currency_matches(
                        extracted_value,
                        expected_value,
                        expected_invoice_currency,
                        extracted_invoice_currency
                    ):
                        correct += 1
                    else:
                        errors.append({
                            'entity': 'items',
                            'field': f'item_{item_num}.{field}',
                            'expected': expected_value,
                            'extracted': extracted_value,
                            'match': False
                        })
                else:
                    if self._values_match(extracted_value, expected_value, field=field):
                        correct += 1
                    else:
                        errors.append({
                            'entity': 'items',
                            'field': f'item_{item_num}.{field}',
                            'expected': expected_value,
                            'extracted': extracted_value,
                            'match': False
                        })

        accuracy = (correct / total * 100) if total > 0 else 100.0
        return {'accuracy': accuracy, 'errors': errors}

    def _currency_matches(
        self,
        extracted: str,
        expected: str,
        expected_invoice_currency: str,
        extracted_invoice_currency: str
    ) -> bool:
        extracted_str = "" if extracted is None else str(extracted).strip()
        expected_str = "" if expected is None else str(expected).strip()
        if not expected_str:
            return True
        if extracted_str:
            return self._values_match(extracted_str, expected_str)
        if expected_str and expected_invoice_currency and expected_str == expected_invoice_currency:
            if extracted_invoice_currency and self._values_match(extracted_invoice_currency, expected_str):
                return True
        return False

    def _values_match(self, extracted: str, expected: str, field: Optional[str] = None) -> bool:
        """Check if two text values match (with normalization and field-specific rules)"""
        extracted_str = "" if extracted is None else str(extracted).strip()
        expected_str = "" if expected is None else str(expected).strip()

        if not extracted_str and not expected_str:
            return True
        if not extracted_str or not expected_str:
            return False

        if field in {'issue_date', 'fulfillment_date', 'due_date'}:
            return self._normalize_date(extracted_str) == self._normalize_date(expected_str)

        if field == 'phone':
            return self._normalize_phone(extracted_str) == self._normalize_phone(expected_str)

        ext_norm = self._normalize_text(extracted_str)
        exp_norm = self._normalize_text(expected_str)

        # Exact match after normalization
        if ext_norm == exp_norm:
            return True

        # For addresses and names, allow partial matches
        # (OCR may miss accents or have slight variations)
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, ext_norm, exp_norm).ratio()
        return similarity >= 0.85  # 85% similarity threshold

    def _numbers_match(self, extracted: str, expected: str, tolerance: float = 0.01) -> bool:
        """Check if two numeric values match (with tolerance)"""
        try:
            ext_val = float(str(extracted).replace(',', '.'))
            exp_val = float(str(expected).replace(',', '.'))

            # Allow 1% tolerance for rounding differences
            diff = abs(ext_val - exp_val)
            threshold = abs(exp_val) * tolerance

            return diff <= threshold
        except (ValueError, TypeError):
            # If can't convert to number, fall back to text matching
            return self._values_match(extracted, expected)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison (lowercase, remove accents, trim)"""
        import unicodedata

        if not text:
            return ''

        # Convert to string if not already
        text = str(text)

        # Normalize unicode (remove accents)
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(c for c in text if not unicodedata.combining(c))

        # Lowercase and trim
        text = text.lower().strip()

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text

    def _normalize_date(self, value: str) -> str:
        if not value:
            return ""

        value = str(value).strip()
        formats = [
            "%Y.%m.%d",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y %m %d",
            "%d.%m.%Y",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d %m %Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        normalized = value.replace('.', '-').replace('/', '-').replace(' ', '-')
        while '--' in normalized:
            normalized = normalized.replace('--', '-')
        return normalized

    def _normalize_phone(self, value: str) -> str:
        import re

        if not value:
            return ""

        digits = re.sub(r"\D", "", str(value))
        if not digits:
            return ""

        canonical = digits
        if canonical.startswith("00") and len(canonical) > 2:
            canonical = canonical[2:]
        if canonical.startswith("06") and len(canonical) > 2:
            canonical = "36" + canonical[2:]
        if canonical.startswith("36"):
            return canonical
        return canonical

    def _generate_summary(self):
        """Generate and display summary statistics"""
        logger.info("=" * 80)
        logger.info("SEMANTIC ACCURACY SUMMARY")
        logger.info("=" * 80)

        if not self.results:
            logger.warning("No results to summarize")
            return

        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']

        if successful:
            avg_semantic = sum(r['semantic_accuracy'] for r in successful) / len(successful)
            avg_seller = sum(r['seller_accuracy'] for r in successful) / len(successful)
            avg_buyer = sum(r['buyer_accuracy'] for r in successful) / len(successful)
            avg_metadata = sum(r['metadata_accuracy'] for r in successful) / len(successful)
            avg_items = sum(r['items_accuracy'] for r in successful) / len(successful)

            logger.info(f"Success Rate: {len(successful)}/{len(self.results)} ({len(successful)/len(self.results)*100:.1f}%)")
            logger.info("")
            logger.info(f"Average Semantic Accuracy: {avg_semantic:.1f}%")
            logger.info(f"  - Seller: {avg_seller:.1f}%")
            logger.info(f"  - Buyer: {avg_buyer:.1f}%")
            logger.info(f"  - Metadata: {avg_metadata:.1f}%")
            logger.info(f"  - Items: {avg_items:.1f}%")
            logger.info("")

            # Show per-invoice results
            logger.info("Per-Invoice Results:")
            for r in successful:
                status_icon = "✅" if r['semantic_accuracy'] >= 95.0 else "❌"
                logger.info(f"  {status_icon} {r['filename']}: {r['semantic_accuracy']:.1f}%")

            # Check if 95% threshold met
            below_threshold = [r for r in successful if r['semantic_accuracy'] < 95.0]
            if below_threshold:
                logger.warning("")
                logger.warning(f"⚠️  {len(below_threshold)} invoices below 95% semantic accuracy:")
                for r in below_threshold:
                    logger.warning(f"  - {r['filename']}: {r['semantic_accuracy']:.1f}%")

                    # Show top 5 field errors
                    if r['field_errors']:
                        logger.warning(f"    Top errors:")
                        for err in r['field_errors'][:5]:
                            logger.warning(f"      • {err['entity']}.{err['field']}: expected '{err['expected']}' got '{err['extracted']}'")
            else:
                logger.info("")
                logger.info("🎉 All invoices meet 95% semantic accuracy threshold!")

        if failed:
            logger.error("")
            logger.error(f"❌ Failed: {len(failed)} invoices")
            for r in failed:
                logger.error(f"  - {r['filename']}: {r['errors']}")

        logger.info("=" * 80)

    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"semantic_results_{timestamp}.json"

        output_data = {
            'timestamp': timestamp,
            'config': {
                'model': self.config.OLLAMA_MODEL,
                'gpu_layers': self.config.OLLAMA_NUM_GPU,
                'ocr_dpi': self.config.OCR_DPI
            },
            'results': self.results
        }

        with output_file.open('w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {output_file}")


def main():
    """Main entry point"""
    invoice_dir = INVOICE_DIR
    ground_truth_path = GROUND_TRUTH_DEFAULT

    if not invoice_dir.exists():
        logger.error(f"Invoice directory not found: {invoice_dir}")
        return

    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return

    tester = SemanticAccuracyTester(invoice_dir, ground_truth_path)
    tester.run_semantic_tests()


if __name__ == '__main__':
    main()
