"""
Step 1: Categorize invoices based on text quality for thesis testing.

Categories:
1. Poor quality - corrupted OCR, high garbage character ratio, unreadable text
2. Medium quality - some OCR artifacts, moderately clean text extraction
3. High quality - clean text, minimal garbage characters, well-formed extraction

Quality metrics:
- Garbage character ratio: special symbols like ^$*\{}[]@#~`
- Clean character ratio: alphanumeric characters (including Hungarian)

Usage:
    python thesis/1_categorize.py

The script will load categorization from thesis_output/invoice_categories.json
If the file doesn't exist, it will auto-categorize all PDFs and create the file.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image
import io

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define invoice category structure (3 categories based on text quality)
CATEGORY_STRUCTURE = {
    "category_1_poor_quality": {
        "description": "Poor text quality - corrupted OCR, high garbage character ratio, unreadable text",
        "invoices": []
    },
    "category_2_medium_quality": {
        "description": "Medium text quality - some OCR artifacts, moderately clean text extraction",
        "invoices": []
    },
    "category_3_high_quality": {
        "description": "High text quality - clean text, minimal garbage characters, well-formed extraction",
        "invoices": []
    }
}

def analyze_pdf(pdf_path):
    """Analyze PDF to determine category based on text quality metrics"""
    try:
        pdf = pdfium.PdfDocument(pdf_path)

        # Extract text from first 3 pages
        text = ""
        for page_num in range(min(3, len(pdf))):
            page = pdf[page_num]
            textpage = page.get_textpage()
            text += textpage.get_text_range()

        text = text.strip()

        # If very little text extracted, categorize as poor
        if len(text) < 50:
            return "category_1_poor_quality"

        # Calculate text quality metrics
        # Garbage characters: OCR artifacts and special symbols
        garbage_chars = len(re.findall(r'[\^\$\*\\\{\}\[\]@#~`]', text))
        # Clean characters: alphanumeric (including Hungarian)
        clean_chars = len(re.findall(r'[a-zA-Z0-9áéíóöőúüűÁÉÍÓÖŐÚÜŰ]', text))
        total_chars = len(text)

        # Detect digit substitution (numbers replacing accented characters)
        # Count words that mix letters and digits (e.g., "Felel6ss6gtl", "V5rmegyei")
        words = re.findall(r'\b\w+\b', text)
        mixed_words = [w for w in words if re.search(r'[a-zA-ZáéíóöőúüűÁÉÍÓÖŐÚÜŰ]', w) and re.search(r'\d', w)]
        digit_substitution_ratio = len(mixed_words) / len(words) if len(words) > 0 else 0

        garbage_ratio = garbage_chars / total_chars if total_chars > 0 else 0
        clean_ratio = clean_chars / total_chars if total_chars > 0 else 0

        # Categorize based on text quality ratios
        # Poor quality: High garbage, low clean ratio, or digit substitution
        if garbage_ratio >= 0.02 or clean_ratio < 0.50 or digit_substitution_ratio > 0.15:
            return "category_1_poor_quality"
        # High quality: Minimal garbage, high clean ratio
        elif garbage_ratio < 0.005 and clean_ratio > 0.65:
            return "category_3_high_quality"
        # Medium quality: Some garbage, acceptable clean ratio
        else:
            return "category_2_medium_quality"

    except Exception as e:
        logger.warning(f"Error analyzing {pdf_path.name}: {e}")
        return "category_2_medium_quality"  # Default fallback

def auto_categorize_pdfs(invoice_dir):
    """Automatically categorize all PDFs in invoice_templates"""
    logger.info("Starting automatic PDF categorization...")

    categories = {k: v.copy() for k, v in CATEGORY_STRUCTURE.items()}
    all_pdfs = sorted(invoice_dir.glob("*.pdf"))

    for i, pdf_path in enumerate(all_pdfs, 1):
        category = analyze_pdf(pdf_path)
        categories[category]["invoices"].append(pdf_path.name)

        if i % 50 == 0:
            logger.info(f"Processed {i}/{len(all_pdfs)} PDFs...")

    logger.info(f"Categorization complete! Processed {len(all_pdfs)} PDFs")
    return categories

def load_or_create_categorization():
    """Load categorization from JSON file or create by auto-categorizing"""
    output_dir = Path(__file__).resolve().parent.parent / "thesis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    categories_file = output_dir / "invoice_categories.json"
    invoice_dir = Path(__file__).parent / "invoice_templates"

    # Get all PDF files from invoice_templates
    all_pdfs = sorted([f.name for f in invoice_dir.glob("*.pdf")])

    if categories_file.exists():
        logger.info(f"Loading categorization from {categories_file}")
        with categories_file.open('r', encoding='utf-8') as f:
            categories = json.load(f)
        return categories
    else:
        logger.warning(f"Categorization file not found: {categories_file}")
        logger.info(f"Will auto-categorize {len(all_pdfs)} PDFs from invoice_templates")

        # Automatically categorize PDFs
        categories = auto_categorize_pdfs(invoice_dir)

        # Save categorization directly to invoice_categories.json
        with categories_file.open('w', encoding='utf-8') as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)

        logger.info(f"Categorization saved to {categories_file}")
        return categories

CATEGORIES = None

def verify_files(categories):
    """Verify all categorized files exist"""
    invoice_dir = Path(__file__).parent / "invoice_templates"

    all_files = set()
    for category, data in categories.items():
        all_files.update(data["invoices"])

    existing_files = {f.name for f in invoice_dir.glob("*.pdf")}

    print(f"Categorized files: {len(all_files)}")
    print(f"Existing files: {len(existing_files)}")

    missing = all_files - existing_files
    extra = existing_files - all_files

    if missing:
        print(f"\n⚠️  Missing files in invoice_templates: {missing}")
    if extra:
        print(f"\n⚠️  Extra files not categorized ({len(extra)}): {sorted(list(extra))[:10]}{'...' if len(extra) > 10 else ''}")

    return len(missing) == 0, len(extra)

def save_categories(categories):
    """Save categories to JSON file"""
    output_dir = Path(__file__).resolve().parent.parent / "thesis_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "invoice_categories.json"

    with output_file.open('w', encoding='utf-8') as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)

    logger.info(f"Categories saved to {output_file}")

def print_summary(categories):
    """Print category summary"""
    logger.info("\n" + "="*80)
    logger.info("INVOICE CATEGORIES")
    logger.info("="*80)

    total_invoices = 0
    for category, data in categories.items():
        count = len(data['invoices'])
        total_invoices += count
        logger.info(f"\n{category}:")
        logger.info(f"  Description: {data['description']}")
        logger.info(f"  Count: {count}")
        for invoice in data['invoices']:
            logger.info(f"    - {invoice}")

    logger.info(f"\nTotal invoices categorized: {total_invoices}")


def main():
    """Main entry point."""
    logger.info("="*80)
    logger.info("STEP 1: CATEGORIZE INVOICES")
    logger.info("="*80)

    # Load categorization
    categories = load_or_create_categorization()

    if categories is None:
        logger.error("Categorization failed")
        return 1

    print_summary(categories)

    all_valid, extra_count = verify_files(categories)

    if all_valid:
        logger.info("\nAll categorized files exist in invoice_templates")
        if extra_count > 0:
            logger.warning(f"Note: {extra_count} PDF files in invoice_templates are not categorized")
        save_categories(categories)
        logger.info("Categories saved successfully")
        return 0
    else:
        logger.error("\nFile verification failed - some categorized files are missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
