# Invoice AI Privacy - On-Premise Processing

Privacy-focused invoice processing using Pytesseract OCR and Ollama LLM. Complete alternative to cloud-based AI services.

## 🔒 Privacy Features

- **Zero external API calls** - everything runs on-premise
- **No data leakage** - invoices never leave your infrastructure
- **GDPR compliant** - full data sovereignty
- **Open source** - transparent and auditable

## 🏗️ Architecture

```
PDF Invoice → Pytesseract OCR → Raw Text → Ollama LLM → Structured JSON
```

**Single Docker Container:**
- Flask API server
- Pytesseract for OCR processing
- Ollama with Llama 2 7B model
- Compatible API interface

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM (16GB recommended)
- ARM64 or x86_64 architecture

### Local Development
```bash
git clone https://github.com/your-username/invoice-ai-privacy.git
cd invoice-ai-privacy
docker-compose up --build
```

### Oracle Cloud Deployment
```bash
# Run setup script
./deployment/oracle_setup.sh

# Build and deploy
docker build -t invoice-ai-privacy .
docker run -p 5000:5000 invoice-ai-privacy
```

## 📋 API Documentation

### Process Invoice
```
POST /process-invoice
Content-Type: multipart/form-data

Body:
- file: PDF file (required)
- processor: "privacy" (optional)

Response: Same JSON structure as OpenAI version
```

### Health Check
```
GET /health
Response: {"status": "healthy", "model": "loaded"}
```

## 🔧 Configuration

Environment variables:
```bash
OLLAMA_HOST=localhost:11434
MODEL_NAME=llama2:7b
OCR_LANGUAGE=eng
DEBUG=false
PORT=5000
```

## 📊 Performance

**Typical Processing Times:**
- OCR (Pytesseract): 2-5 seconds
- LLM (Llama 2 7B): 10-30 seconds
- Total: 15-35 seconds per invoice

**Resource Requirements:**
- RAM: 6-8GB during processing
- CPU: 2+ cores recommended
- Storage: 10GB+ (models + temp files)

## 🆚 vs OpenAI Version

| Feature | OpenAI | Privacy |
|---------|--------|---------|
| Data Privacy | ❌ External | ✅ On-premise |
| Cost | $0.10-0.50/invoice | Free (hosting only) |
| Speed | 2-5 seconds | 15-35 seconds |
| Accuracy | High | Good (depends on model) |
| Scalability | Unlimited | Hardware limited |

## 🛠️ Development

### Project Structure
```
├── app.py                 # Flask main application
├── config.py              # Configuration management
├── utils/
│   ├── ocr.py            # Pytesseract wrapper
│   ├── llm.py            # Ollama client
│   └── processing.py     # Main pipeline
├── models/
│   └── download_models.sh
├── tests/
│   └── test_api.py
└── deployment/
    ├── oracle_setup.sh
    └── docker-compose.oracle.yml
```

### Testing
```bash
# Unit tests
python -m pytest tests/

# Test with sample invoice
curl -X POST -F "file=@tests/sample_invoices/test.pdf" \
  http://localhost:5000/process-invoice
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📞 Support

- Issues: GitHub Issues
- Documentation: /docs folder
- Performance tuning: See deployment guide