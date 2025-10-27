# Invoice AI Privacy

A completely local invoice processing service using Ollama for AI-powered data extraction. No cloud dependencies, full privacy.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Thesis Evaluation Pipeline](#thesis-evaluation-pipeline)
- [API Usage](#api-usage)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Features

- **Complete Privacy**: Everything runs locally, no data leaves your machine
- **Fast Processing**: Local LLM with GPU acceleration
- **Real-time Progress**: Live progress tracking with cancellation support
- **PDF Support**: OCR-based text extraction from PDF invoices using docTR
- **Secure API**: Bearer token authentication for production use
- **One-Click Setup**: Automated startup script handles everything
- **High Accuracy**: Semantic accuracy validation with ground truth comparison
- **Smart Validation**: Automatic quantity and price validation with OCR error correction

## Quick Start

### Prerequisites

```bash
# Install Ollama
# Windows: Download from https://ollama.ai/download/windows
# Linux/Mac: curl -fsSL https://ollama.com/install.sh | sh

# Verify installation
ollama --version

# Install Python 3.9+
python --version
```

### Setup

```bash
# Clone repository
git clone <your-repo>
cd invoice-ai-privacy

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env and set your API_KEY
```

### Launch

```bash
# Start Ollama service (if not already running)
ollama serve

# Pull model (first time only)
ollama pull llama3.1:8b-instruct-q6_K

# Start Flask API
python app.py
```

## Thesis Evaluation Pipeline

The thesis evaluation pipeline provides automated testing and benchmarking for invoice extraction accuracy.

### Directory Structure

```
thesis/
├── invoice_templates/     # Place your test invoice PDFs here (gitignored)
│   └── .gitkeep
├── 1_categorize.py       # Categorize invoices by quality
├── 2_generate_ground_truth.py  # Generate ground truth with OpenAI Vision
├── 3_semantic_check.py   # Run LLM extraction and semantic accuracy check
├── 4_export_csv.py       # Export results to CSV
└── main_evaluation.py    # Run complete pipeline

thesis_output/
├── invoice_categories.json              # Invoice categories
├── ground_truth_vision.json             # OpenAI Vision ground truth
├── semantic_results_YYYYMMDD_HHMMSS.json  # Detailed results
└── category_benchmark_YYYYMMDD_HHMMSS.csv # Aggregated CSV
```

### Setup Test Invoices

```bash
# Place your test PDF invoices in thesis/invoice_templates/
cp /path/to/your/invoices/*.pdf thesis/invoice_templates/

# Update categories in thesis/1_categorize.py if needed
```

### Run Complete Pipeline

```bash
# Run all steps in sequence
python thesis/main_evaluation.py

# This will:
# 1. Categorize invoices by quality
# 2. Generate ground truth (OpenAI Vision) - only if missing
# 3. Run semantic accuracy check (Local LLM)
# 4. Export results to CSV
```

### Run Individual Steps

```bash
# Step 1: Categorize invoices
python thesis/1_categorize.py

# Step 2: Generate ground truth (requires OpenAI API key)
# Set OPENAI_API_KEY in environment or in ../invoice-ai/.env.local
python thesis/2_generate_ground_truth.py

# Step 3: Run semantic accuracy check
python thesis/3_semantic_check.py

# Step 4: Export CSV
python thesis/4_export_csv.py
```

### CSV Output Format

The generated CSV contains three sections:

1. **Overall Summary**: Total count and overall statistics
2. **Category Summary**: Aggregated statistics per category
3. **Detailed Results**: Per-invoice accuracy and timing

Categories:
- `category_1_poor_scan`: Poor scan quality - low DPI, skewed, blurry
- `category_2_medium_scan`: Medium scan quality - readable but OCR challenges
- `category_3_good_scan`: Good scan quality - clear scanned document
- `category_4_digital_image`: Digital invoice (non-searchable) - image-based PDF
- `category_5_digital_text`: Digital e-invoice (searchable) - born-digital PDF

## API Usage

### Health Check

```bash
curl http://localhost:5000/health
```

### Process Invoice

```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@invoice.pdf" \
  http://localhost:5000/process-invoice
```

### Supported Invoice Formats

- PDF files (single or multi-page)
- Hungarian and English invoices
- Standard invoice layouts
- Two-column layouts (seller/buyer side-by-side)

## Configuration

Edit `.env` file:

```env
# Security
API_KEY=your_secure_api_key_here

# Ollama Settings
OLLAMA_MODEL=llama3.1:8b-instruct-q6_K
OLLAMA_HOST=localhost:11434

# GPU Optimization (adjust based on your GPU)
# RTX 2060 SUPER (8GB): 40 recommended
# RTX 3060 (12GB): 45+
# RTX 3070 (8GB): 42-45
# RTX 4060 Ti (16GB): 50+
# CPU only: 0
OLLAMA_NUM_GPU=40

# Processing
MAX_FILE_SIZE=52428800  # 50MB
OCR_LANGUAGE=hun+eng
OCR_DPI=300  # Higher DPI = better accuracy but slower

# Server
PORT=5000
HOST=0.0.0.0
DEBUG=false
```

### Performance Tuning

**Optimal Settings for RTX 2060 SUPER (8GB):**
- `OLLAMA_NUM_GPU=40` (40 layers on GPU, rest on CPU)
- `num_thread=8` (CPU threads for remaining layers)
- `num_batch=512` (batch size for prompt processing)
- `OCR_DPI=300` (balance between speed and accuracy)

**Expected Performance:**
- Metadata extraction: 20-35 seconds
- Items extraction: 70-110 seconds
- Total per invoice: 90-145 seconds

**VRAM Usage:**
- 40 layers: 5-6GB VRAM
- 42 layers: 5.5-6.5GB VRAM
- 45 layers: 6-7GB VRAM (maximum safe)

**Optimization Tips:**
- Monitor VRAM: `nvidia-smi`
- Close GPU-intensive applications
- Adjust `OLLAMA_NUM_GPU` based on available VRAM

## Architecture

```
invoice-ai-privacy/
├── app.py              # Flask API server
├── config.py           # Configuration management
├── utils/
│   ├── processing.py   # Main processing pipeline
│   ├── llm.py          # Ollama client
│   └── ocr.py          # PDF/OCR processing
├── thesis/             # Evaluation pipeline
│   ├── invoice_templates/  # Test invoices (gitignored)
│   ├── 1_categorize.py
│   ├── 2_generate_ground_truth.py
│   ├── 3_semantic_check.py
│   ├── 4_export_csv.py
│   └── main_evaluation.py
├── thesis_output/      # Generated results (gitignored)
└── requirements.txt    # Python dependencies
```

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Manual development server
python app.py
```

## Testing

### Semantic Accuracy Tests

```bash
# Run complete evaluation pipeline
python thesis/main_evaluation.py

# Results saved to:
# - thesis_output/semantic_results_[timestamp].json
# - thesis_output/category_benchmark_[timestamp].csv
```

### Ground Truth Dataset

The `thesis_output/ground_truth_vision.json` file contains expected values for test invoices generated using OpenAI Vision API.

### Accuracy Metrics

The semantic accuracy check compares extracted data against ground truth across:
- **Seller data**: Name, address, tax ID, email, phone
- **Buyer data**: Name, address, tax ID
- **Metadata**: Invoice number, dates, payment method, currency
- **Line items**: Product names, quantities, unit prices, net/gross amounts

Results include:
- Overall accuracy (weighted average)
- Per-category statistics (mean, standard deviation)
- Per-invoice detailed results

## Troubleshooting

### Ollama Issues

```bash
# Check if running
# Windows:
tasklist | findstr ollama
# Linux/Mac:
ps aux | grep ollama

# Manual start
ollama serve

# Check models
ollama list

# Pull model if missing
ollama pull llama3.1:8b-instruct-q6_K
```

### Python Issues

```bash
# Recreate virtual environment
# Windows:
rmdir /s venv
# Linux/Mac:
rm -rf venv

python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### Port Conflicts

- Change `PORT=5000` in `.env` if port is in use
- Check with: `netstat -ano | findstr :5000` (Windows) or `lsof -i :5000` (Linux/Mac)

### OpenAI Vision Issues (Step 2)

Ground truth generation requires OpenAI API key:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY=your_api_key

# Option 2: Add to ../invoice-ai/.env.local
OPENAI_API_KEY=your_api_key
```

## Security Notes

- Never commit `.env` - Contains sensitive API keys
- Generate secure API keys for production
- Log files may contain sensitive data
- All invoice PDFs in `thesis/invoice_templates/` are gitignored
- Ground truth and results in `thesis_output/` are gitignored

## License

Private repository - All rights reserved
