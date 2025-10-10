# Invoice AI Privacy

A completely local invoice processing service using Ollama (Gemma2 9B) for AI-powered data extraction. No cloud dependencies, full privacy.

**Latest Update (2025-10-09)**: Improved semantic accuracy from 84.9% to 90.7% average through structured OCR and validation fixes. See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for details.

## Features

- **🔒 Complete Privacy**: Everything runs locally, no data leaves your machine
- **⚡ Fast Processing**: Gemma2 9B model with multilingual support (Hungarian/English)
- **📊 Real-time Progress**: Live progress tracking with cancellation support
- **📄 PDF Support**: OCR-based text extraction from PDF invoices using docTR
- **🔑 Secure API**: Bearer token authentication for production use
- **🚀 One-Click Setup**: Automated startup script handles everything
- **✅ High Accuracy**: 90.7% average semantic accuracy with validation and correction
- **🛠️ Smart Validation**: Automatic quantity and price validation with OCR error correction

## Quick Start

1. **Prerequisites**
   ```bash
   # Install Ollama
   # Windows: Download from https://ollama.ai/download/windows
   # Verify installation
   ollama --version

   # Install Python 3.9+
   python --version
   ```

2. **Setup Environment**
   ```bash
   # Clone and setup
   git clone <your-repo>
   cd invoice-ai-privacy

   # Copy environment template
   copy .env.example .env
   # Edit .env and set your API_KEY
   ```

3. **Install Everything (First Time)**
   ```bash
   # Right-click and "Run as Administrator"
   install.bat

   # This will install:
   # - Python 3.11
   # - Ollama AI runtime
   # - Git and ngrok
   # - Download Qwen2.5 3B Instruct Q4 model (~2GB)
   # - Setup Python virtual environment
   ```

4. **Launch the Service**
   ```bash
   # Regular run (after installation)
   launch.bat

   # The launcher will:
   # - Start Ollama service
   # - Verify model availability
   # - Start ngrok tunnel (optional)
   # - Launch Flask API
   ```

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

**Supported Invoice Formats:**
- PDF files (single or multi-page)
- Hungarian and English invoices
- Standard invoice layouts (tested with auto parts, telecom, service invoices)
- Two-column layouts (seller/buyer side-by-side)

**Note**: PDFs are processed using docTR OCR engine (GPU-accelerated). Ensure test PDFs are placed in a directory accessible to the backend (e.g., `C:/Users/[username]/Downloads/invoice_templates/`).

## Configuration

Edit `.env` file:

```env
# Security
API_KEY=your_secure_api_key_here

# Ollama Settings
OLLAMA_MODEL=gemma2:9b
OLLAMA_HOST=localhost:11434

# GPU Optimization (adjust based on your GPU):
# RTX 2060 SUPER (8GB): OLLAMA_NUM_GPU=40-45 (recommended: 40)
# RTX 3060 (12GB): OLLAMA_NUM_GPU=45+
# RTX 3070 (8GB): OLLAMA_NUM_GPU=42-45
# RTX 4060 Ti (16GB): OLLAMA_NUM_GPU=50+
# CPU only: OLLAMA_NUM_GPU=0
OLLAMA_NUM_GPU=40

# Processing
MAX_FILE_SIZE=52428800  # 50MB
OCR_LANGUAGE=hun+eng
OCR_DPI=300  # Higher DPI = better OCR accuracy but slower

# Server
PORT=5000
HOST=0.0.0.0
DEBUG=false
```

### Performance Tuning for RTX 2060 SUPER (8GB)

The backend is optimized for RTX 2060 SUPER with 8GB VRAM:

**Optimal Settings:**
- `OLLAMA_NUM_GPU=40` (loads 40 layers on GPU, rest on CPU)
- `num_thread=8` (8 CPU threads for remaining layers)
- `num_batch=512` (batch size for faster prompt processing)
- `OCR_DPI=300` (balance between speed and accuracy)

**Expected Performance:**
- Metadata extraction: ~20-35 seconds
- Items extraction: ~70-110 seconds
- Total per invoice: ~90-145 seconds (20-30% faster than 30 layers)

**VRAM Usage:**
- 40 layers: ~5-6GB VRAM
- 42 layers: ~5.5-6.5GB VRAM
- 45 layers: ~6-7GB VRAM (maximum safe)

**To Optimize Further:**
- Increase to `OLLAMA_NUM_GPU=42-45` if you have headroom (check with `nvidia-smi`)
- Close other GPU-intensive applications (browsers, games)
- Ensure Ollama has GPU access: `nvidia-smi` should show ollama process
- Monitor VRAM during processing: `nvidia-smi dmon -s u`

## Architecture

```
├── app.py              # Flask API server
├── config.py           # Configuration management
├── utils/
│   ├── processing.py   # Main processing pipeline
│   ├── llm.py         # Ollama client
│   └── ocr.py         # PDF/OCR processing
├── start-native.bat   # Windows startup script
└── requirements.txt   # Python dependencies
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

## Model Information

- **Gemma2 9B**: High-quality model with excellent accuracy for structured data extraction
- **GPU Accelerated**: Optimized for RTX 2060+ GPUs with 6GB+ VRAM (30 GPU layers)
- **Privacy**: Models downloaded locally, no external API calls
- **Cancellation**: Real-time processing cancellation with streaming responses
- **Performance**: ~120 seconds per invoice with chunked extraction
- **Accuracy**: 90.7% average semantic accuracy with validation and correction

## Testing

### Run Semantic Accuracy Tests

```bash
# Test with ground truth validation
python tests/semantic_accuracy_test.py

# Results saved to tests/semantic_results_[timestamp].json
```

### Ground Truth Dataset

The `tests/ground_truth.json` file contains expected values for test invoices:
- `real_HEB04803.pdf` - Hungarian auto parts invoice
- `generated_invoice_001.pdf` - Generated multi-item invoice
- `real_2025_3428303_e1.pdf` - Telecom invoice

### Accuracy Metrics

Tests measure semantic accuracy across:
- **Seller data**: Name, address, tax ID, contact info (100% accuracy)
- **Buyer data**: Name, address, tax ID (72.2% accuracy)
- **Metadata**: Invoice number, dates, payment method (100% accuracy)
- **Line items**: Product names, quantities, prices (90.6% accuracy)

See [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for detailed accuracy analysis.

## Troubleshooting

### Ollama Issues
```bash
# Check if running
tasklist | findstr ollama

# Manual start
ollama serve

# Check models
ollama list
```

### Python Issues
```bash
# Recreate virtual environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Port Conflicts
- Change `PORT=5000` in `.env` if port 5000 is in use
- Update ngrok command in startup script accordingly

## Security Notes

- ⚠️ **Never commit `.env`** - Contains sensitive API keys
- 🔒 **Generate secure API keys** for production
- 🌐 **ngrok tunnels** are public - use strong authentication
- 📝 **Log files** may contain sensitive data

## License

Private repository - All rights reserved