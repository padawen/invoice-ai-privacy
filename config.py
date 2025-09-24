import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration class for Invoice AI Privacy service"""

    # Flask settings
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')

    # Ollama settings
    OLLAMA_HOST = os.getenv('OLLAMA_HOST', 'localhost:11434')
    OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama2:7b')
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', 120))  # 2 minutes

    # OCR settings
    OCR_LANGUAGE = os.getenv('OCR_LANGUAGE', 'eng')
    OCR_DPI = int(os.getenv('OCR_DPI', 300))
    OCR_CONFIG = '--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6

    # Processing settings
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 50 * 1024 * 1024))  # 50MB
    ALLOWED_EXTENSIONS = {'pdf'}
    TEMP_DIR = os.getenv('TEMP_DIR', '/tmp/invoice_processing')

    # Model settings
    MODELS_DIR = os.getenv('MODELS_DIR', './models')

    # API settings
    API_KEY = os.getenv('API_KEY', None)  # Optional API key for security

    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    @staticmethod
    def init_app(app):
        """Initialize app with config"""
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)

class DevelopmentConfig(Config):
    DEBUG = True
    OLLAMA_MODEL = 'phi3:mini'  # Smaller model for development

class ProductionConfig(Config):
    DEBUG = False
    OLLAMA_MODEL = 'llama2:7b'  # Full model for production

class TestingConfig(Config):
    TESTING = True
    OLLAMA_MODEL = 'phi3:mini'  # Fast model for testing

# Config mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}