"""
Category-based benchmark test for thesis evaluation.
Tests invoices by quality category and computes statistics per category.

Output format requested by thesis advisor (András):
- Categories: poor_scan, medium_scan, good_scan, digital_image, digital_text
- Metrics per category: avg_accuracy, std_accuracy, avg_time, std_time
- Output: Large CSV table
"""

import json
import logging
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils.processing import InvoiceProcessor

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "thesis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_PATH = OUTPUT_DIR / "invoice_categories.json"
GROUND_TRUTH_PATH = PROJECT_ROOT / "thesis_output" / "ground_truth_vision.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_categories():
    """Load invoice categories"""
    if not CATEGORIES_PATH.exists():
        raise FileNotFoundError(f"Invoice category file not found: {CATEGORIES_PATH}")
    with CATEGORIES_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_ground_truth():
    """Load ground truth data"""
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(f"Ground truth file not found: {GROUND_TRUTH_PATH}")
    with GROUND_TRUTH_PATH.open('r', encoding='utf-8') as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    Normalize text for semantic comparison:
    - Remove accents/diacritics
    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation
    """
    import unicodedata
    import re

    if not text:
        return ""

    # Convert to string
    text = str(text)

    # Remove accents (ó -> o, ű -> u, etc)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    # Lowercase
    text = text.lower()

    # Remove punctuation except numbers and letters
    text = re.sub(r'[^\w\s]', '', text)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text


def normalize_number(value: str) -> float:
    """
    Normalize numeric value to float for comparison.
    Handles: "1.234,56", "1,234.56", "1234.56", "1234"
    """
    import re

    if not value:
        return 0.0

    # Convert to string
    value = str(value).strip()

    # Remove currency symbols and spaces
    value = re.sub(r'[^\d,.-]', '', value)

    if not value:
        return 0.0

    # European format: 1.234,56 -> 1234.56
    if ',' in value and '.' in value:
        if value.rfind(',') > value.rfind('.'):
            # European: 1.234,56
            value = value.replace('.', '').replace(',', '.')
        else:
            # US: 1,234.56
            value = value.replace(',', '')
    elif ',' in value:
        # Could be decimal (1,5) or thousands (1,234)
        if value.count(',') == 1 and len(value.split(',')[1]) <= 2:
            # Decimal: 1,5 -> 1.5
            value = value.replace(',', '.')
        else:
            # Thousands: 1,234 -> 1234
            value = value.replace(',', '')

    try:
        return float(value)
    except:
        return 0.0


def normalize_date(date_str: str) -> str:
    """
    Normalize date to canonical YYYY-MM-DD format.
    Handles variations like "2024.12.04", "13/02/2025", "2025 02 13".
    """
    if not date_str:
        return ""

    date_str = str(date_str).strip()

    candidates = [
        "%Y.%m.%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y %m %d",
        "%d.%m.%Y",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d %m %Y",
    ]

    for fmt in candidates:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    normalized = date_str.replace('.', '-').replace('/', '-').replace(' ', '-')
    while '--' in normalized:
        normalized = normalized.replace('--', '-')
    return normalized


def normalize_phone(value: str) -> str:
    """Normalize phone numbers for semantic comparison."""
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


def semantic_match(extracted_value, ground_truth_value, field_type='text') -> bool:
    """
    Semantically compare two values based on field type.

    Args:
        extracted_value: Value from extraction
        ground_truth_value: Ground truth value
        field_type: 'text', 'number', 'date'

    Returns:
        True if values match semantically
    """
    from difflib import SequenceMatcher

    # Handle empty/None values
    if not extracted_value and not ground_truth_value:
        return True
    if not extracted_value or not ground_truth_value:
        return False

    if field_type == 'number':
        # Numeric comparison with tolerance
        try:
            ex_num = normalize_number(extracted_value)
            gt_num = normalize_number(ground_truth_value)

            # Allow 0.1% difference for rounding
            tolerance = max(abs(gt_num * 0.001), 0.01)
            return abs(ex_num - gt_num) <= tolerance
        except:
            return False

    elif field_type == 'date':
        # Date comparison (normalized)
        ex_date = normalize_date(extracted_value)
        gt_date = normalize_date(ground_truth_value)
        return ex_date == gt_date

    elif field_type == 'phone':
        ex_phone = normalize_phone(extracted_value)
        gt_phone = normalize_phone(ground_truth_value)
        if not ex_phone or not gt_phone:
            return ex_phone == gt_phone
        return ex_phone == gt_phone

    else:  # text
        # Text comparison with fuzzy matching
        ex_norm = normalize_text(extracted_value)
        gt_norm = normalize_text(ground_truth_value)

        # Exact match after normalization
        if ex_norm == gt_norm:
            return True

        # Fuzzy match - allow 85% similarity
        similarity = SequenceMatcher(None, ex_norm, gt_norm).ratio()
        return similarity >= 0.85


def calculate_accuracy(extracted: dict, ground_truth: dict) -> float:
    """
    Calculate semantic accuracy between extracted and ground truth data.
    Uses intelligent matching: normalizes text, numbers, dates.

    Returns accuracy as percentage (0-100).
    """
    total_fields = 0
    correct_fields = 0

    # Compare seller fields
    for field in ['name', 'address', 'tax_id', 'email', 'phone']:
        if field in ground_truth.get('seller', {}):
            total_fields += 1
            ex_val = extracted.get('seller', {}).get(field, '')
            gt_val = ground_truth['seller'][field]
            field_type = 'phone' if field == 'phone' else 'text'
            if semantic_match(ex_val, gt_val, field_type=field_type):
                correct_fields += 1

    # Compare buyer fields
    for field in ['name', 'address', 'tax_id']:
        if field in ground_truth.get('buyer', {}):
            total_fields += 1
            ex_val = extracted.get('buyer', {}).get(field, '')
            gt_val = ground_truth['buyer'][field]
            if semantic_match(ex_val, gt_val, field_type='text'):
                correct_fields += 1

    # Compare metadata fields
    date_fields = ['issue_date', 'fulfillment_date', 'due_date']
    for field in ['invoice_number', 'issue_date', 'fulfillment_date', 'due_date', 'payment_method', 'currency']:
        if field in ground_truth:
            total_fields += 1
            ex_val = extracted.get(field, '')
            gt_val = ground_truth[field]
            field_type = 'date' if field in date_fields else 'text'
            if semantic_match(ex_val, gt_val, field_type=field_type):
                correct_fields += 1

    # Compare invoice items
    gt_items = ground_truth.get('invoice_data', [])
    ex_items = extracted.get('invoice_data', [])

    # Get overall invoice currency for fallback
    invoice_currency = extracted.get('currency', '') or ground_truth.get('currency', '')

    for i, gt_item in enumerate(gt_items):
        ex_item = ex_items[i] if i < len(ex_items) else {}

        for field in ['name', 'quantity', 'unit_price', 'net', 'gross', 'currency']:
            if field in gt_item:
                total_fields += 1
                ex_val = ex_item.get(field, '')
                gt_val = gt_item[field]

                # Special handling for currency: fall back to invoice-level currency when missing
                if field == 'currency' and not ex_val and invoice_currency:
                    if semantic_match(invoice_currency, gt_val, field_type='text'):
                        correct_fields += 1
                        continue

                # Determine field type
                if field in ['quantity', 'unit_price', 'net', 'gross']:
                    field_type = 'number'
                else:
                    field_type = 'text'

                if semantic_match(ex_val, gt_val, field_type=field_type):
                    correct_fields += 1

    return (correct_fields / total_fields * 100) if total_fields > 0 else 0


def test_invoice(invoice_path: Path, ground_truth: dict, config: Config) -> Dict:
    """Test a single invoice and return results"""
    logger.info(f"Testing: {invoice_path.name}")

    start_time = time.time()

    try:
        # Process invoice
        with open(invoice_path, 'rb') as f:
            pdf_bytes = f.read()

        # Create processor and process invoice
        processor = InvoiceProcessor(config)
        result = processor.process_pdf(pdf_bytes, filename=invoice_path.name)

        processing_time = time.time() - start_time

        # Calculate accuracy
        accuracy = calculate_accuracy(result, ground_truth)

        # Extract timing metadata if available
        metadata = result.get('_processing_metadata', {})

        return {
            'filename': invoice_path.name,
            'status': 'success',
            'accuracy': accuracy,
            'processing_time': processing_time,
            'ocr_duration': metadata.get('ocr_duration', 0),
            'llm_duration': metadata.get('llm_duration', 0)
        }

    except Exception as e:
        logger.error(f"Error processing {invoice_path.name}: {e}")
        return {
            'filename': invoice_path.name,
            'status': 'error',
            'error': str(e),
            'accuracy': 0,
            'processing_time': time.time() - start_time
        }


def run_category_benchmark():
    """Run benchmark tests by category"""
    logger.info("="*80)
    logger.info("CATEGORY-BASED BENCHMARK TEST")
    logger.info("="*80)

    # Load data
    categories = load_categories()
    ground_truth = load_ground_truth()
    config = Config()

    invoice_dir = Path("C:/Users/Davide/Downloads/invoice_templates")

    # Results per category
    category_results = {}

    # Test each category
    for category_id, category_data in categories.items():
        logger.info(f"\nTesting {category_id}: {category_data['description']}")
        logger.info(f"Invoices: {len(category_data['invoices'])}")

        if len(category_data['invoices']) == 0:
            logger.info("Skipping empty category")
            continue

        results = []

        for invoice_filename in category_data['invoices']:
            invoice_path = invoice_dir / invoice_filename

            if not invoice_path.exists():
                logger.warning(f"Invoice not found: {invoice_path}")
                continue

            # Check if ground truth exists
            if invoice_filename not in ground_truth:
                logger.warning(f"No ground truth for {invoice_filename} - skipping")
                continue

            result = test_invoice(invoice_path, ground_truth[invoice_filename], config)
            results.append(result)

        # Calculate statistics
        if results:
            accuracies = [r['accuracy'] for r in results if r['status'] == 'success']
            times = [r['processing_time'] for r in results if r['status'] == 'success']

            category_results[category_id] = {
                'description': category_data['description'],
                'count': len(results),
                'avg_accuracy': statistics.mean(accuracies) if accuracies else 0,
                'std_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
                'avg_time': statistics.mean(times) if times else 0,
                'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                'results': results
            }

    return category_results


def export_to_csv(category_results: Dict, output_file: str):
    """Export results to CSV as requested by advisor"""
    import csv

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'Category',
            'Description',
            'Count',
            'Avg Accuracy (%)',
            'Std Accuracy',
            'Avg Time (s)',
            'Std Time (s)'
        ])

        # Data rows
        for category_id, data in category_results.items():
            writer.writerow([
                category_id,
                data['description'],
                data['count'],
                f"{data['avg_accuracy']:.2f}",
                f"{data['std_accuracy']:.2f}",
                f"{data['avg_time']:.2f}",
                f"{data['std_time']:.2f}"
            ])

    logger.info(f"\nResults exported to {output_file}")


def print_summary(category_results: Dict):
    """Print summary table"""
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("="*80)

    for category_id, data in category_results.items():
        logger.info(f"\n{category_id}:")
        logger.info(f"  Description: {data['description']}")
        logger.info(f"  Count: {data['count']}")
        logger.info(f"  Avg Accuracy: {data['avg_accuracy']:.2f}% (std: {data['std_accuracy']:.2f})")
        logger.info(f"  Avg Time: {data['avg_time']:.2f}s (std: {data['std_time']:.2f}s)")


if __name__ == "__main__":
    results = run_category_benchmark()
    print_summary(results)
    export_to_csv(results, str(OUTPUT_DIR / 'category_benchmark_results.csv'))
