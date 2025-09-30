# Invoice AI Privacy

A completely local invoice processing service using Ollama (Qwen2.5 7B Instruct Q4) for AI-powered data extraction. No cloud dependencies, full privacy.

## Features

- **🔒 Complete Privacy**: Everything runs locally, no data leaves your machine
- **⚡ Fast Processing**: Mistral 7B Instruct model with multilingual support (Hungarian/English)
- **📊 Real-time Progress**: Live progress tracking with cancellation support
- **📄 PDF Support**: OCR-based text extraction from PDF invoices
- **🔑 Secure API**: Bearer token authentication for production use
- **🚀 One-Click Setup**: Automated startup script handles everything

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
   # - Download Qwen2.5 7B Instruct Q4 model (~4GB)
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

## Configuration

Edit `.env` file:

```env
# Security
API_KEY=your_secure_api_key_here

# Ollama Settings
OLLAMA_MODEL=llama3.2:3b
OLLAMA_HOST=localhost:11434

# Processing
MAX_FILE_SIZE=52428800  # 50MB
OCR_LANGUAGE=eng

# Server
PORT=5000
HOST=0.0.0.0
DEBUG=false
```

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

- **Qwen2.5 7B Instruct Q4**: Optimal balance of speed and accuracy for invoice processing
- **GPU Accelerated**: Optimized for RTX 2070+ GPUs with 4-6GB VRAM
- **Privacy**: Models downloaded locally, no external API calls
- **Cancellation**: Real-time processing cancellation with streaming responses
- **Performance**: 3-6 second response times on GPU, excellent JSON output quality

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