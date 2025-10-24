"""
Utility script to keep the local invoice dataset in sync.

Steps performed:
1. Categorise every PDF in the invoice_templates directory that is not yet
   listed in thesis_output/invoice_categories.json. A simple heuristic assigns
   invoices with an embedded text layer to the digital_text category and
   everything else to the digital_image bucket.
2. Generate ground-truth entries in thesis_output/ground_truth_vision.json for
   invoices that do not yet have them by calling the same OpenAI Vision
   pipeline (prompt + formatting rules) used by the Next.js app.
3. Execute the semantic accuracy evaluation so results stay current.

Run from the repository root:
    python thesis/dataset_sync.py
"""

import base64
import json
import logging
import os
import re
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: openai. Install with 'pip install openai'.") from exc

try:
    from playwright.sync_api import sync_playwright
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: playwright. Install with 'pip install playwright' and run 'playwright install chromium'."
    ) from exc

# Ensure repository modules resolve for downstream imports
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(Path(__file__).parent.resolve()) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.resolve()))

from config import Config  # noqa: E402
from utils.processing import InvoiceProcessor  # noqa: E402
from semantic_accuracy_eval import SemanticAccuracyTester  # noqa: E402

OUTPUT_DIR = ROOT_DIR / "thesis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INVOICE_DIR = Path("C:/Users/Davide/Downloads/invoice_templates")
CATEGORIES_PATH = OUTPUT_DIR / "invoice_categories.json"
GROUND_TRUTH_PATH = ROOT_DIR / "thesis_output" / "ground_truth_vision.json"
INVOICE_AI_ENV_FILE = ROOT_DIR.parent / "invoice-ai" / ".env.local"
INSTRUCTIONS_FILE = ROOT_DIR.parent / "invoice-ai" / "lib" / "instructions.ts"
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL") or "gpt-4o"


def load_env_local() -> Dict[str, str]:
    """Parse invoice-ai/.env.local into a dict if available."""
    env: Dict[str, str] = {}
    if not INVOICE_AI_ENV_FILE.exists():
        return env

    for line in INVOICE_AI_ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def load_guidelines_text() -> str:
    """
    Extract the image guidelines text from the Next.js app so we reuse the exact prompt.
    """
    if not INSTRUCTIONS_FILE.exists():
        raise RuntimeError(f"Instructions file not found: {INSTRUCTIONS_FILE}")

    content = INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    func_index = content.find("export function getGuidelinesImage(): string")
    if func_index == -1:
        raise RuntimeError("Failed to locate getGuidelinesImage definition.")

    return_index = content.find("return `", func_index)
    if return_index == -1:
        raise RuntimeError("Failed to locate return statement for getGuidelinesImage.")

    start = return_index + len("return `")
    end = content.find("`", start)
    if end == -1:
        raise RuntimeError("Failed to locate closing backtick in getGuidelinesImage.")

    return content[start:end].strip()


_env_local = load_env_local()
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or _env_local.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_KEY")
    or _env_local.get("OPENAI_KEY")
)

if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is not configured. Set env or add to invoice-ai/.env.local")
    sys.exit(1)

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
GUIDELINES_TEXT = load_guidelines_text()

DEFAULT_DESCRIPTIONS = {
    "category_3_good_scan": "Good scan quality - clear scanned document, good OCR",
    "category_4_digital_image": "Digital invoice (non-searchable) - clean but image-based PDF",
    "category_5_digital_text": "Digital e-invoice (searchable) - born-digital, text-based PDF",
}


def load_json(path: Path, default) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(default))  # deep copy


def ensure_category_structure(data: dict) -> dict:
    """Ensure category dict contains the expected keys and descriptions."""
    for key, description in DEFAULT_DESCRIPTIONS.items():
        entry = data.setdefault(key, {})
        entry.setdefault("description", description)
        entry.setdefault("invoices", [])
    return data


def detect_category(pdf_path: Path, processor: InvoiceProcessor) -> str:
    """
    Heuristic categorisation:
      - If the PDF has a sizeable embedded text layer, treat as digital_text.
      - Otherwise default to digital_image (non-searchable).  This keeps the
        process automatic; manual promotion to good_scan can be done later.
    """
    try:
        native = processor._extract_pdf_native_text(pdf_path.read_bytes())  # type: ignore[attr-defined]
        if native and native.get("char_count", 0) > 200:
            return "category_5_digital_text"
    except Exception as exc:  # pragma: no cover
        logging.warning("Failed to inspect %s for embedded text: %s", pdf_path.name, exc)

    return "category_4_digital_image"


def update_categories(invoice_files: List[str]) -> None:
    categories = ensure_category_structure(load_json(CATEGORIES_PATH, {}))

    assigned = {name for cat in categories.values() for name in cat.get("invoices", [])}
    missing = sorted(set(invoice_files) - assigned)

    if not missing:
        logging.info("All invoices already categorised.")
        return

    logging.info("Categorising %d new invoice(s).", len(missing))
    processor = InvoiceProcessor(Config())

    for filename in missing:
        category = detect_category(INVOICE_DIR / filename, processor)
        invoices = categories[category]["invoices"]
        if filename not in invoices:
            invoices.append(filename)
            logging.info("  %s -> %s", filename, category)

    for entry in categories.values():
        entry["invoices"] = sorted(set(entry.get("invoices", [])))

    with open(CATEGORIES_PATH, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2, ensure_ascii=False)
    logging.info("Updated categories written to %s", CATEGORIES_PATH)


def clean_ground_truth_entry(entry: dict) -> dict:
    """Strip volatile fields before persisting ground truth."""
    entry = dict(entry)
    entry.pop("_processing_metadata", None)
    entry.pop("id", None)
    return entry


def pdf_to_base64_images(pdf_path: Path) -> List[str]:
    """Convert PDF pages to base64-encoded PNG strings using the same rendering approach as the Next.js app."""
    temp_dir = Path(tempfile.mkdtemp(prefix="invoice-pages-"))
    image_paths: List[Path] = []

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-accelerated-2d-canvas",
                    "--no-first-run",
                    "--no-zygote",
                    "--disable-gpu",
                ],
            )
            context = browser.new_context(viewport={"width": 1200, "height": 1600})
            page = context.new_page()

            base64_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
            encoded_pdf = json.dumps(base64_pdf)
            html = f"""
            <!DOCTYPE html>
            <html>
              <head>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
                <style>
                  body {{ margin: 0; padding: 20px; background: white; }}
                  #pdf-container {{ width: 100%; }}
                  canvas {{ display: block; margin: 20px 0; border: 1px solid #ccc; }}
                </style>
              </head>
              <body>
                <div id="pdf-container"></div>
                <script>
                  pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                  async function renderPDF() {{
                    try {{
                      const pdfData = atob({encoded_pdf});
                      const pdf = await pdfjsLib.getDocument({{ data: pdfData }}).promise;
                      const container = document.getElementById('pdf-container');
                      for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {{
                        const page = await pdf.getPage(pageNum);
                        const viewport = page.getViewport({{ scale: 2.0 }});
                        const canvas = document.createElement('canvas');
                        canvas.width = viewport.width;
                        canvas.height = viewport.height;
                        canvas.id = 'page-' + pageNum;
                        const context = canvas.getContext('2d');
                        await page.render({{ canvasContext: context, viewport }}).promise;
                        container.appendChild(canvas);
                      }}
                      window.pdfRendered = true;
                    }} catch (error) {{
                      console.error('PDF rendering error:', error);
                      window.pdfError = error?.message || String(error);
                    }}
                  }}
                  renderPDF();
                </script>
              </body>
            </html>
            """

            page.set_content(html, wait_until="networkidle")
            page.wait_for_function("() => window.pdfRendered || window.pdfError", timeout=30_000)
            pdf_error = page.evaluate("() => window.pdfError || null")
            if pdf_error:
                raise RuntimeError(f"PDF rendering failed: {pdf_error}")

            canvases = page.query_selector_all("canvas")
            if not canvases:
                raise RuntimeError("No canvases rendered from PDF; cannot generate images.")

            for index, canvas in enumerate(canvases, start=1):
                output_path = temp_dir / f"page-{index}.png"
                canvas.screenshot(path=str(output_path))
                if not output_path.exists() or output_path.stat().st_size == 0:
                    raise RuntimeError(f"Rendered image missing or empty: {output_path}")
                image_paths.append(output_path)

            context.close()
            browser.close()

        encoded_images = [base64.b64encode(path.read_bytes()).decode("utf-8") for path in image_paths]
        return encoded_images
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def call_openai_vision(pdf_path: Path) -> Optional[dict]:
    """Send invoice to OpenAI Vision API and parse structured response."""
    try:
        images_b64 = pdf_to_base64_images(pdf_path)
    except Exception as exc:
        logging.error("  %s -> failed to convert PDF to images: %s", pdf_path.name, exc)
        return None

    content = [{"type": "text", "text": GUIDELINES_TEXT}]

    for image_b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=4096,
            temperature=0,
        )
    except Exception as exc:
        logging.error("  %s -> OpenAI API error: %s", pdf_path.name, exc)
        return None

    text = response.choices[0].message.content.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logging.error("  %s -> failed to parse JSON response: %s", pdf_path.name, exc)
        return None

    return data


def generate_ground_truth(pdf_path: Path) -> Optional[dict]:
    """Generate ground truth strictly via OpenAI Vision API."""
    result = call_openai_vision(pdf_path)
    if result and result.get("invoice_data"):
        logging.info(
            "  %s -> extracted %d items via OpenAI Vision",
            pdf_path.name,
            len(result["invoice_data"]),
        )
        return clean_ground_truth_entry(result)

    logging.error("  %s -> OpenAI Vision returned no data", pdf_path.name)
    return None


def update_ground_truth(invoice_files: List[str]) -> None:
    ground_truth = load_json(GROUND_TRUTH_PATH, {})
    missing = sorted(set(invoice_files) - set(ground_truth.keys()))
    if not missing:
        logging.info("Ground truth already up to date.")
        return

    logging.info("Generating ground truth for %d invoice(s).", len(missing))

    updated = False
    for filename in missing:
        pdf_path = INVOICE_DIR / filename
        if not pdf_path.exists():
            logging.warning("  Skipping missing file: %s", pdf_path)
            continue

        entry = generate_ground_truth(pdf_path)
        if entry:
            ground_truth[filename] = entry
            updated = True
        else:
            logging.error("  Failed to extract ground truth for %s", filename)

        # Save incrementally to avoid losing progress on failure
        if updated:
            with open(GROUND_TRUTH_PATH, "w", encoding="utf-8") as f:
                json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    if updated:
        logging.info("Ground truth saved to %s", GROUND_TRUTH_PATH)
    else:
        logging.info("No ground truth changes written.")


def run_semantic_tests() -> None:
    if not GROUND_TRUTH_PATH.exists():
        logging.warning("Ground truth file not found; skipping semantic accuracy tests.")
        return

    logging.info("Running semantic accuracy tests...")
    tester = SemanticAccuracyTester(str(INVOICE_DIR), str(GROUND_TRUTH_PATH))
    tester.run_semantic_tests()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    run_comparison = "--compare" in sys.argv

    if not INVOICE_DIR.exists():
        logging.error("Invoice directory not found: %s", INVOICE_DIR)
        sys.exit(1)

    invoice_files = sorted(p.name for p in INVOICE_DIR.glob("*.pdf"))
    if not invoice_files:
        logging.warning("No PDF invoices found in %s", INVOICE_DIR)
        return

    logging.info("Processing %d invoice(s).", len(invoice_files))
    update_categories(invoice_files)
    update_ground_truth(invoice_files)
    if run_comparison:
        run_semantic_tests()
    else:
        logging.info("Skipping semantic accuracy tests (no '--compare' flag). Run once with '--compare' when ready.")


if __name__ == "__main__":
    main()
