from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create directories
for dir_path in [OUTPUT_DIR / "images", OUTPUT_DIR / "crops", OUTPUT_DIR / "results"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Detection settings
CONFIDENCE_THRESHOLD = 0.68

# Education keywords for filtering
EDUCATION_KEYWORDS = [
    'school', 'schools', 'education', 'educational', 'student', 'students',
    'teacher', 'teachers', 'university', 'college', 'academic', 'learning',
    'classroom', 'curriculum', 'exam', 'exams', 'graduation', 'degree',
    'tuition', 'scholarship', 'principal', 'kindergarten', 'elementary',
    'secondary', 'admission', 'enrollment', 'faculty', 'campus', 'study',
    'studies', 'homework', 'textbook', 'library', 'semester', 'grade'
]

# Filtering settings
KEYWORD_MIN_MATCH = 2

# OCR settings
OCR_LANG = "eng"
OCR_PSM_PRIMARY = 6
OCR_PSM_FALLBACK = 4

# Detection/Extraction tweaks
BBOX_PADDING_PCT = 0.03  # expand bbox by 3% on each side before OCR

# Concurrency
NUM_WORKERS = 4

# LLM/Summarization settings
# Default to a lighter summarization model for speed and memory; can be overridden via CLI
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_SUMMARY_LENGTH = 150
MAX_INPUT_CHARS_FOR_SUMMARY = 2000
