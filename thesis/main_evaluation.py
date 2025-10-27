"""
Main Thesis Evaluation Pipeline Orchestrator

Runs the complete evaluation workflow in order:
1. Categorize invoices (1_categorize.py)
2. Generate ground truth with OpenAI Vision (2_generate_ground_truth.py) - only if missing
3. Run semantic check with local LLM (3_semantic_check.py)
4. Export results to CSV (4_export_csv.py)

Usage:
    python thesis/main_evaluation.py
"""

import sys
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Setup paths
ROOT_DIR = Path(__file__).parent.parent.resolve()
THESIS_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = ROOT_DIR / "thesis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES_PATH = OUTPUT_DIR / "invoice_categories.json"
GROUND_TRUTH_PATH = OUTPUT_DIR / "ground_truth_vision.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_step(step_name: str, script_path: Path, description: str) -> bool:
    """
    Run a pipeline step (Python script).

    Returns True if successful, False otherwise.
    """
    logger.info("\n" + "="*80)
    logger.info(f"{step_name}: {description}")
    logger.info("="*80)

    try:
        # Use sys.executable to ensure same Python interpreter
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            check=True,
            capture_output=False,  # Show output in real-time
        )
        logger.info(f"{step_name} completed successfully.")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"{step_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"{step_name} failed: {e}")
        return False


def should_run_ground_truth() -> bool:
    """
    Check if ground truth generation should run.

    Returns True if:
    - ground_truth_vision.json doesn't exist, OR
    - There are new invoices without ground truth
    """
    if not GROUND_TRUTH_PATH.exists():
        logger.info("Ground truth file not found - will generate")
        return True

    # Check for new invoices
    import json
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    invoice_dir = THESIS_DIR / "invoice_templates"
    if not invoice_dir.exists():
        return False

    invoice_files = set(p.name for p in invoice_dir.glob("*.pdf"))
    ground_truth_files = set(ground_truth.keys())

    new_invoices = invoice_files - ground_truth_files

    if new_invoices:
        logger.info(f"Found {len(new_invoices)} new invoice(s) without ground truth")
        return True

    logger.info("Ground truth is up to date - skipping generation")
    return False


def main():
    """Main evaluation pipeline."""
    logger.info("\n" + "="*80)
    logger.info("THESIS EVALUATION PIPELINE - FULL RUN")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*80)

    start_time = datetime.now()

    # Step 1: Categorize
    step1_script = THESIS_DIR / "1_categorize.py"
    if not run_step("STEP 1", step1_script, "Categorize invoices"):
        logger.error("Pipeline aborted at Step 1")
        return 1

    # Step 2: Ground Truth (conditional)
    if should_run_ground_truth():
        step2_script = THESIS_DIR / "2_generate_ground_truth.py"
        if not run_step("STEP 2", step2_script, "Generate ground truth (OpenAI Vision)"):
            logger.error("Pipeline aborted at Step 2")
            return 1
    else:
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Ground truth generation - SKIPPED (up to date)")
        logger.info("="*80)

    # Step 3: Semantic Check
    step3_script = THESIS_DIR / "3_semantic_check.py"
    if not run_step("STEP 3", step3_script, "Semantic accuracy check (Local LLM)"):
        logger.error("Pipeline aborted at Step 3")
        return 1

    # Step 4: Export CSV
    step4_script = THESIS_DIR / "4_export_csv.py"
    if not run_step("STEP 4", step4_script, "Export results to CSV"):
        logger.error("Pipeline aborted at Step 4")
        return 1

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Total elapsed time: {elapsed}")
    logger.info(f"Results directory: {OUTPUT_DIR}")
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  - {CATEGORIES_PATH.name}")
    logger.info(f"  - {GROUND_TRUTH_PATH.name}")
    logger.info(f"  - semantic_results_YYYYMMDD_HHMMSS.json (latest)")
    logger.info(f"  - category_benchmark_YYYYMMDD_HHMMSS.csv (latest)")
    logger.info("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
