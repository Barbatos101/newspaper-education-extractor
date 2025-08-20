from pathlib import Path
import os
import requests
import logging

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "bestmodel.pt"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create necessary directories
for d in [OUTPUT_DIR / "images", OUTPUT_DIR / "crops", OUTPUT_DIR / "results"]:
    d.mkdir(parents=True, exist_ok=True)

def ensure_model_downloaded():
    """Download model from GitHub Release if not present"""
    if not MODEL_PATH.exists():
        print("Model not found. Downloading from GitHub Release...")
        # Updated URL for your repo
        url = "https://github.com/Barbatos101/newspaper-education-extractor/releases/download/v1.0/bestmodel.pt"
        
        try:
            MODEL_PATH.parent.mkdir(exist_ok=True)
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            with open(MODEL_PATH, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {percent:.1f}%", end='')
            print(f"\nModel downloaded successfully to {MODEL_PATH}")
        except Exception as e:
            logging.error(f"Failed to download model: {e}")
            raise

# Download model on import
ensure_model_downloaded()

# Cloud Run optimized settings
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.68'))
KEYWORD_MIN_MATCH = int(os.getenv('KEYWORD_MIN_MATCH', '2'))

# Education keywords
EDUCATION_KEYWORDS = [
    'school', 'schools', 'education', 'educational',
    'student', 'students', 'teacher', 'teachers',
    'university', 'college', 'academic', 'classroom',
    'curriculum', 'exam', 'exams', 'graduation',
    'scholarship', 'principal', 'kindergarten', 'elementary',
    'secondary', 'admission', 'enrollment', 'faculty', 'campus',
    'homework', 'textbook', 'library', 'semester', 'grade',
    'syllabus', 'tuition', 'institute', 'staff', 'board'
]

# OCR settings
OCR_LANG = "eng"
OCR_PSM_PRIMARY = 6
OCR_PSM_FALLBACK = 4

# Threading - optimized for Cloud Run
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '1'))

# Summarization settings
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_SUMMARY_LENGTH = int(os.getenv('MAX_SUMMARY_LENGTH', '120'))
MAX_INPUT_CHARS_FOR_SUMMARY = int(os.getenv('MAX_INPUT_CHARS_FOR_SUMMARY', '1200'))

# Performance optimization for Cloud Run
REDUCED_DPI = int(os.getenv('REDUCED_DPI', '150'))
SEMANTIC_THRESHOLD = 0.35
