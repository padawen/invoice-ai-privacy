#!/bin/bash

# Model download script for Invoice AI Privacy

set -e

echo "=== Invoice AI Privacy Model Setup ==="

# Check if Ollama is running
if ! curl -f http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "❌ Ollama is not running. Please start Ollama first."
    exit 1
fi

echo "✅ Ollama is running"

# Default models to download
DEVELOPMENT_MODEL="phi3:mini"      # ~2GB, fast for development
PRODUCTION_MODEL="llama2:7b"       # ~4GB, better accuracy

# Check environment or use default
MODEL_NAME=${OLLAMA_MODEL:-$DEVELOPMENT_MODEL}

echo "📦 Downloading model: $MODEL_NAME"

# Pull the model
if ollama pull "$MODEL_NAME"; then
    echo "✅ Model $MODEL_NAME downloaded successfully!"
else
    echo "❌ Failed to download model $MODEL_NAME"
    exit 1
fi

# Verify model is available
echo "📋 Available models:"
ollama list

echo ""
echo "🎉 Model setup complete!"
echo "   Model: $MODEL_NAME"
echo "   Usage: Set OLLAMA_MODEL environment variable to use different models"
echo ""
echo "Available models for Invoice AI:"
echo "  - phi3:mini     (~2GB) - Fast, good for development"
echo "  - llama2:7b     (~4GB) - Better accuracy, production ready"
echo "  - mistral:7b    (~4GB) - Alternative high-quality model"
echo "  - codellama:7b  (~4GB) - Good for structured data extraction"