from pathlib import Path
import os

# Set tokenizers parallelism to false to prevent fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODEL_PATH = PROJECT_ROOT / "models" / "bestmodel.pt"
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Cloud Run optimizations
NUM_WORKERS = int(os.getenv('NUM_WORKERS', '1'))
REDUCED_DPI = int(os.getenv('REDUCED_DPI', '150'))
CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.78'))
MAX_INPUT_CHARS_FOR_SUMMARY = int(os.getenv('MAX_SUMMARY_CHARS', '900'))
KEYWORD_MIN_MATCH = int(os.getenv('KEYWORD_MIN_MATCH', '2'))

# Create directories
for dir_path in [OUTPUT_DIR / "images", OUTPUT_DIR / "crops", OUTPUT_DIR / "results"]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Detection settings
CONFIDENCE_THRESHOLD = 0.78

# Refined education keywords - strictly school/education focused
EDUCATION_KEYWORDS = [
    'school', 'schools', 'education', 'educational',
    'student', 'students', 'teacher', 'teachers',
    'university', 'college', 'academic', 'classroom',
    'curriculum', 'exam', 'exams', 'graduation',
    'scholarship', 'principal', 'kindergarten', 'elementary',
    'secondary', 'admission', 'enrollment', 'faculty', 'campus',
    'homework', 'textbook', 'library', 'semester', 'grade',
    'syllabus', 'tuition', 'institute', 'staff', 'board',
    'classroom', 'lesson', 'lessons', 'instructor', 'pupil',
    'pupils', 'academy', 'preschool', 'high school', 'primary'
]

# Filtering settings
KEYWORD_MIN_MATCH = 2

# OCR settings
OCR_LANG = "eng"
OCR_PSM_PRIMARY = 6
OCR_PSM_FALLBACK = 4

# Detection/Extraction tweaks
BBOX_PADDING_PCT = 0.03

# Concurrency - optimized for speed
import platform
if platform.machine() == 'arm64':  # Apple Silicon
    NUM_WORKERS = 4
else:
    NUM_WORKERS = min(6, os.cpu_count())

OCR_THREAD_SAFE = True

# LLM/Summarization settings - using specified model
SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
MAX_SUMMARY_LENGTH = 120
MAX_INPUT_CHARS_FOR_SUMMARY = 1800

# Speed optimization settings
REDUCED_DPI = 220  # Balanced quality/speed
SEMANTIC_THRESHOLD = 0.35  # Threshold for semantic similarity
