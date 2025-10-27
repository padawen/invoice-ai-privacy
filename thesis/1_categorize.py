"""
Step 1: Categorize invoices based on quality for thesis testing.

Categories:
1. Poor scan quality - low DPI, skewed, blurry
2. Medium scan quality - readable but some OCR challenges
3. Good scan quality - clear scanned document
4. Digital invoice (non-searchable) - clean but image-based PDF
5. Digital e-invoice (searchable) - born-digital, text-based PDF

Usage:
    python thesis/1_categorize.py
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define invoice categories manually after inspection
CATEGORIES = {
    "category_1_poor_scan": {
        "description": "Poor scan quality - low DPI, skewed, blurry, OCR difficult",
        "invoices": []
    },
    "category_2_medium_scan": {
        "description": "Medium scan quality - readable but some OCR challenges",
        "invoices": []
    },
    "category_3_good_scan": {
        "description": "Good scan quality - clear scanned document, good OCR",
        "invoices": [
            "real_invoice_1.pdf",  # DuóVill - 3 pages, complex table, good quality
            "real_invoice_2.pdf"   # DuóVill - 1 page, simpler, good quality
        ]
    },
    "category_4_digital_image": {
        "description": "Digital invoice (non-searchable) - clean but image-based PDF",
        "invoices": [
            "real_invoice_3.pdf",  # Large file (807K) suggests high-res image
            "real_invoice_4.pdf"   # Medium file (76K)
        ]
    },
    "category_5_digital_text": {
        "description": "Digital e-invoice (searchable) - born-digital, text-based PDF",
        "invoices": [
            "real_2025_3428303_e1.pdf",  # E-invoice from filename
            "real_HEB04803.pdf",
            "real_invoice-1.pdf",
            "real_invoice-2.pdf",
            "real_invoice-3.pdf",
            "real_W_2025_4642.pdf"
        ]
    }
}

def verify_files():
    """Verify all categorized files exist"""
    invoice_dir = Path(__file__).parent / "invoice_templates"

    all_files = set()
    for category, data in CATEGORIES.items():
        all_files.update(data["invoices"])

    existing_files = {f.name for f in invoice_dir.glob("real_*.pdf")}

    print(f"Categorized files: {len(all_files)}")
    print(f"Existing files: {len(existing_files)}")

    missing = all_files - existing_files
    extra = existing_files - all_files

    if missing:
        print(f"\n⚠️  Missing files: {missing}")
    if extra:
        print(f"\n⚠️  Extra files not categorized: {extra}")

    return len(missing) == 0 and len(extra) == 0

def save_categories():
    """Save categories to JSON file"""
    output_dir = Path(__file__).resolve().parent.parent / "thesis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "invoice_categories.json"

    with output_file.open('w', encoding='utf-8') as f:
        json.dump(CATEGORIES, f, indent=2, ensure_ascii=False)

    print(f"\nOK - Categories saved to {output_file}")

def print_summary():
    """Print category summary"""
    logger.info("\n" + "="*80)
    logger.info("INVOICE CATEGORIES")
    logger.info("="*80)

    for category, data in CATEGORIES.items():
        logger.info(f"\n{category}:")
        logger.info(f"  Description: {data['description']}")
        logger.info(f"  Count: {len(data['invoices'])}")
        for invoice in data['invoices']:
            logger.info(f"    - {invoice}")


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("STEP 1: CATEGORIZE INVOICES")
    logger.info("="*80)

    print_summary()

    if verify_files():
        logger.info("\nAll files verified")
        save_categories()
        logger.info("Categories saved successfully")
        return 0
    else:
        logger.error("\nFile verification failed - please review categories")
        return 1


if __name__ == "__main__":
    sys.exit(main())
