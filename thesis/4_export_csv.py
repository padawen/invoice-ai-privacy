"""
Export semantic check results to CSV format.

This script:
- Loads semantic check results (JSON)
- Loads invoice categories
- Aggregates results by category (mean, std)
- Exports to timestamped CSV

Usage:
    python thesis/4_export_csv.py                           # Use latest semantic_results_*.json
    python thesis/4_export_csv.py --input results.json      # Use specific JSON file
"""

import json
import csv
import sys
import logging
import statistics
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Setup paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_DIR = ROOT_DIR / "thesis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CATEGORIES_PATH = OUTPUT_DIR / "invoice_categories.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_latest_semantic_results() -> Optional[Path]:
    """Find the most recent semantic_results_*.json file."""
    results_files = sorted(OUTPUT_DIR.glob("semantic_results_*.json"), reverse=True)
    if results_files:
        return results_files[0]
    return None


def load_json(path: Path) -> dict:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_categories() -> dict:
    """Load invoice categories."""
    if not CATEGORIES_PATH.exists():
        logger.error(f"Categories file not found: {CATEGORIES_PATH}")
        logger.error("Run 1_categorize.py first to generate categories.")
        sys.exit(1)

    return load_json(CATEGORIES_PATH)


def aggregate_by_category(semantic_results: dict, categories: dict) -> Dict:
    """
    Aggregate semantic check results by category.

    Returns:
        {
            'category_id': {
                'description': '...',
                'count': N,
                'avg_accuracy': X.XX,
                'std_accuracy': X.XX,
                'avg_time': X.XX,
                'std_time': X.XX
            }
        }
    """
    # Build filename -> category mapping
    filename_to_category = {}
    for category_id, category_data in categories.items():
        for filename in category_data.get('invoices', []):
            filename_to_category[filename] = {
                'id': category_id,
                'description': category_data.get('description', '')
            }

    # Group results by category
    category_groups = {}
    for result in semantic_results.get('results', []):
        filename = result.get('filename', '')
        category_info = filename_to_category.get(filename)

        if not category_info:
            logger.warning(f"No category found for {filename}, skipping")
            continue

        category_id = category_info['id']
        if category_id not in category_groups:
            category_groups[category_id] = {
                'description': category_info['description'],
                'results': []
            }

        category_groups[category_id]['results'].append(result)

    # Calculate statistics per category
    aggregated = {}
    for category_id, data in category_groups.items():
        results = data['results']
        successful = [r for r in results if r.get('status') == 'success']

        if not successful:
            logger.warning(f"No successful results for {category_id}")
            continue

        accuracies = [r.get('semantic_accuracy', 0) for r in successful]
        times = [r.get('total_duration', 0) for r in successful]

        aggregated[category_id] = {
            'description': data['description'],
            'count': len(successful),
            'avg_accuracy': statistics.mean(accuracies) if accuracies else 0,
            'std_accuracy': statistics.stdev(accuracies) if len(accuracies) > 1 else 0,
            'avg_time': statistics.mean(times) if times else 0,
            'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        }

    return aggregated


def export_to_csv(aggregated_results: Dict, semantic_results: dict, categories: dict, output_file: Path):
    """Export aggregated results to CSV with three sections: overall, category summary, and details."""
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')

        # SECTION 0: Overall Summary
        all_results = [r for r in semantic_results.get('results', []) if r.get('status') == 'success']
        if all_results:
            all_accuracies = [r.get('semantic_accuracy', 0) for r in all_results]
            all_times = [r.get('total_duration', 0) for r in all_results]

            writer.writerow(['=== OVERALL SUMMARY ==='])
            writer.writerow([
                'Total Count',
                'Avg Accuracy (%)',
                'Std Accuracy',
                'Avg Time (s)',
                'Std Time (s)'
            ])
            writer.writerow([
                len(all_results),
                f"{statistics.mean(all_accuracies):.2f}".replace('.', ','),
                f"{statistics.stdev(all_accuracies) if len(all_accuracies) > 1 else 0:.2f}".replace('.', ','),
                f"{statistics.mean(all_times):.2f}".replace('.', ','),
                f"{statistics.stdev(all_times) if len(all_times) > 1 else 0:.2f}".replace('.', ',')
            ])
            writer.writerow([])
            writer.writerow([])

        # SECTION 1: Category Summary
        writer.writerow(['=== CATEGORY SUMMARY ==='])
        writer.writerow([
            'Category',
            'Description',
            'Count',
            'Avg Accuracy (%)',
            'Std Accuracy',
            'Avg Time (s)',
            'Std Time (s)'
        ])

        # Summary data rows (sorted by category ID)
        for category_id in sorted(aggregated_results.keys()):
            data = aggregated_results[category_id]
            writer.writerow([
                category_id,
                data['description'],
                data['count'],
                f"{data['avg_accuracy']:.2f}".replace('.', ','),  # Magyar Excel: vessző a tizedesjel
                f"{data['std_accuracy']:.2f}".replace('.', ','),
                f"{data['avg_time']:.2f}".replace('.', ','),
                f"{data['std_time']:.2f}".replace('.', ',')
            ])

        # Empty rows separator
        writer.writerow([])
        writer.writerow([])

        # SECTION 2: Detailed Results
        writer.writerow(['=== DETAILED RESULTS ==='])
        writer.writerow([
            'Category',
            'Filename',
            'Accuracy (%)',
            'Time (s)',
            'Status'
        ])

        # Build filename -> category mapping
        filename_to_category = {}
        for category_id, category_data in categories.items():
            for filename in category_data.get('invoices', []):
                filename_to_category[filename] = category_id

        # Detailed rows (sorted by category, then filename)
        results_by_category = {}
        for result in semantic_results.get('results', []):
            filename = result.get('filename', '')
            category = filename_to_category.get(filename, 'unknown')

            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)

        for category_id in sorted(results_by_category.keys()):
            for result in sorted(results_by_category[category_id], key=lambda r: r.get('filename', '')):
                accuracy = result.get('semantic_accuracy', 0)
                time = result.get('total_duration', 0)
                status = result.get('status', 'unknown')

                writer.writerow([
                    category_id,
                    result.get('filename', ''),
                    f"{accuracy:.2f}".replace('.', ','),
                    f"{time:.2f}".replace('.', ','),
                    status
                ])

    logger.info(f"CSV exported to: {output_file}")


def print_summary(aggregated_results: Dict):
    """Print summary table to console."""
    logger.info("\n" + "="*80)
    logger.info("CATEGORY BENCHMARK SUMMARY")
    logger.info("="*80)

    for category_id in sorted(aggregated_results.keys()):
        data = aggregated_results[category_id]
        logger.info(f"\n{category_id}:")
        logger.info(f"  Description: {data['description']}")
        logger.info(f"  Count: {data['count']}")
        logger.info(f"  Avg Accuracy: {data['avg_accuracy']:.2f}% (std: {data['std_accuracy']:.2f})")
        logger.info(f"  Avg Time: {data['avg_time']:.2f}s (std: {data['std_time']:.2f}s)")

    logger.info("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export semantic check results to CSV by category'
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to semantic_results JSON file (default: latest)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path (default: category_benchmark_TIMESTAMP.csv)'
    )
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("STEP 4: EXPORT CSV")
    logger.info("="*80)

    # Find input JSON
    if args.input:
        input_file = Path(args.input)
        if not input_file.exists():
            logger.error(f"Input file not found: {input_file}")
            sys.exit(1)
    else:
        input_file = find_latest_semantic_results()
        if not input_file:
            logger.error("No semantic_results_*.json files found in thesis_output/")
            logger.error("Run 3_semantic_check.py first to generate results.")
            sys.exit(1)

    logger.info(f"Using input: {input_file}")

    # Load data
    semantic_results = load_json(input_file)
    categories = load_categories()

    # Aggregate by category
    aggregated = aggregate_by_category(semantic_results, categories)

    if not aggregated:
        logger.error("No results to export (no successful extractions or categories)")
        sys.exit(1)

    # Print summary
    print_summary(aggregated)

    # Export CSV
    if args.output:
        output_file = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = OUTPUT_DIR / f"category_benchmark_{timestamp}.csv"

    export_to_csv(aggregated, semantic_results, categories, output_file)

    logger.info("="*80)
    logger.info("CSV export complete.")
    logger.info("="*80)


if __name__ == "__main__":
    main()
