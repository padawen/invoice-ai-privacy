# Invoice AI Privacy

A completely local invoice processing service using Ollama (Mistral 7B Instruct) for AI-powered data extraction. No cloud dependencies, full privacy.

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

3. **Run the Service**
   ```bash
   # Windows: Just double-click or run
   start-native.bat

   # The script will:
   # - Check/start Ollama
   # - Download Mistral 7B Instruct model
   # - Setup Python environment
   # - Start ngrok tunnel
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

- **Mistral 7B Instruct**: Superior multilingual support for Hungarian and English invoices
- **CPU Inference**: Works on Intel/AMD CPUs, no GPU required
- **Privacy**: Models downloaded locally, no external API calls
- **Cancellation**: Real-time processing cancellation with streaming responses

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