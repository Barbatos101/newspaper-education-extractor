import os
import cv2
import fitz
import numpy as np
import pytesseract
import json
import re
import logging
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading

from config import *

# Set environment variables for optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_ocr_text(text):
    """Clean OCR output by removing artifacts"""
    if not text:
        return ""
    
    try:
        # Remove artifacts
        text = re.sub(r'[|\\/=↔_•¤©®™]+', '', text)
        text = re.sub(r'[^\w\s.,!?:;\'\"()\-\n]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Filter meaningful lines
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) > 10 and re.search(r'[a-zA-Z]', line):
                alpha_ratio = len(re.findall(r'[a-zA-Z]', line)) / len(line)
                if alpha_ratio > 0.3:
                    cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines).strip()
    except Exception:
        return str(text).strip() if text else ""

class SemanticEducationFilter:
    """Semantic education filtering using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.sentence_model = SentenceTransformer(model_name)
            self.semantic_available = True
            self.logger.info(f"Loaded semantic model: {model_name}")
        except Exception as e:
            self.logger.error(f"Semantic model loading failed: {e}")
            self.sentence_model = None
            self.semantic_available = False

        self.education_keywords = EDUCATION_KEYWORDS
        
        if self.semantic_available:
            try:
                self.keyword_embeddings = self.sentence_model.encode(self.education_keywords)
            except Exception as e:
                self.logger.error(f"Failed to create embeddings: {e}")
                self.semantic_available = False

    def is_education_article(self, text: str, min_keywords: int = 2) -> Tuple[bool, List[str]]:
        """Determine if text is education-related"""
        
        if not text or len(text.strip()) < 30:
            return False, []
        
        text_lower = text.lower()
        
        # Traditional keyword matching
        found_keywords = []
        for keyword in self.education_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        # Semantic similarity (if available)
        semantic_score = 0.0
        if self.semantic_available and self.sentence_model:
            try:
                text_embedding = self.sentence_model.encode([text])
                keyword_similarities = cosine_similarity(text_embedding, self.keyword_embeddings)
                semantic_score = np.max(keyword_similarities[0])
            except Exception as e:
                self.logger.warning(f"Semantic analysis failed: {e}")
        
        # Decision logic
        keyword_match = len(found_keywords) >= min_keywords
        semantic_match = semantic_score >= SEMANTIC_THRESHOLD
        
        is_education = keyword_match or semantic_match
        return is_education, found_keywords

class NewspaperEducationExtractor:
    def __init__(
        self,
        min_keyword_matches: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        summarization_model: Optional[str] = None,
        num_workers: Optional[int] = None,
        save_crops: bool = False,
    ):
        """Initialize the enhanced extractor"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Runtime settings
        self.keyword_min_match = min_keyword_matches or KEYWORD_MIN_MATCH
        self.confidence_threshold = confidence_threshold or CONFIDENCE_THRESHOLD
        self.num_workers = num_workers or NUM_WORKERS
        self.save_crops = save_crops
        
        # Threading locks
        self._ocr_lock = threading.Lock()
        self._summ_lock = threading.Lock()

        # Load YOLO model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        self.yolo_model = YOLO(str(MODEL_PATH))
        self.education_filter = SemanticEducationFilter()
        
        self.logger.info("Loaded YOLO model and semantic filter")
        
        # Initialize summarization
        try:
            model_name = summarization_model or SUMMARIZATION_MODEL
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                tokenizer=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.logger.info(f"Summarizer loaded: {model_name}")
        except Exception as e:
            self.logger.warning(f"Summarization model failed to load: {e}")
            self.summarizer = None

    def process_newspaper(self, pdf_path: str) -> Dict:
        """Complete processing pipeline"""
        self.logger.info(f"Processing newspaper: {pdf_path}")
        
        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)
        
        # Initialize results
        education_articles = []
        stats = {
            'total_pages': len(image_paths),
            'total_articles_detected': 0,
            'education_articles_found': 0
        }
        
        # Process each page
        for page_num, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing page {page_num}/{len(image_paths)}")
            
            # Detect articles
            articles = self.detect_articles(image_path, page_num)
            stats['total_articles_detected'] += len(articles)
            
            # Process articles with threading
            if articles:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._process_single_article, article): idx
                        for idx, article in enumerate(articles)
                    }
                    
                    for future in as_completed(future_to_idx):
                        try:
                            result = future.result(timeout=120)
                            if result:
                                education_articles.append(result)
                                stats['education_articles_found'] += 1
                                self.logger.info(f"Found education article: Page {result['page']}")
                        except Exception as e:
                            self.logger.warning(f"Article processing error: {e}")
        
        # Return results
        results = {
            'pdf_path': pdf_path,
            'processing_stats': stats,
            'education_articles': education_articles,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'semantic_enabled': self.education_filter.semantic_available,
            'summarization_model': SUMMARIZATION_MODEL
        }
        
        return results

    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[str]:
        """Convert PDF pages to images with optimized DPI"""
        if dpi is None:
            dpi = REDUCED_DPI
            
        self.logger.info(f"Converting PDF at {dpi} DPI")
        
        try:
            pdf_document = fitz.open(pdf_path)
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                mat = fitz.Matrix(dpi/72, dpi/72)
                pix = page.get_pixmap(matrix=mat)
                
                image_filename = f"{pdf_name}_page_{page_num + 1}.png"
                image_path = OUTPUT_DIR / "images" / image_filename
                pix.save(str(image_path))
                image_paths.append(str(image_path))
            
            pdf_document.close()
            return image_paths
        except Exception as e:
            self.logger.error(f"PDF conversion error: {e}")
            return []

    def detect_articles(self, image_path: str, page_num: int) -> List[Dict]:
        """Detect articles using YOLO"""
        try:
            results = self.yolo_model.predict(
                source=image_path,
                conf=self.confidence_threshold,
                imgsz=640,
                verbose=False,
                save=False
            )
            
            articles = []
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    
                    for i, (box, conf) in enumerate(zip(boxes, confidences)):
                        x1, y1, x2, y2 = map(int, box)
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Filter by minimum area
                        if area > 5000:
                            articles.append({
                                'article_id': i + 1,
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'area': area,
                                'image_path': image_path,
                                'page': page_num
                            })
            
            return articles
        except Exception as e:
            self.logger.error(f"Article detection error: {e}")
            return []

    def _process_single_article(self, article: Dict) -> Optional[Dict]:
        """Process one article"""
        try:
            # Extract text using OCR
            crop_path, text = self.extract_article_crop_and_text(article)
            
            if len(text.strip()) < 40:
                return None
            
            # Education filtering
            is_education, keywords = self.education_filter.is_education_article(text, self.keyword_min_match)
            if not is_education:
                return None
            
            # Generate summary
            summary = self.summarize_text(text)
            
            return {
                'page': article['page'],
                'article_id': article['article_id'],
                'confidence': article['confidence'],
                'bbox': article['bbox'],
                'keywords_found': keywords,
                'full_text': text,
                'summary': summary,
                'crop_path': crop_path,
                'text_length': len(text)
            }
        except Exception as e:
            self.logger.error(f"Article processing error: {e}")
            return None

    def extract_article_crop_and_text(self, article_data: Dict) -> Tuple[str, str]:
        """Extract article crop and perform OCR"""
        try:
            img = cv2.imread(article_data['image_path'])
            if img is None:
                return "", ""
            
            x1, y1, x2, y2 = article_data['bbox']
            crop = img[y1:y2, x1:x2]
            
            if crop.size == 0:
                return "", ""

            crop_path_str = ""
            if self.save_crops:
                try:
                    crop_filename = f"article_{article_data['article_id']}_page_{article_data['page']}.jpg"
                    crop_path = OUTPUT_DIR / "crops" / crop_filename
                    cv2.imwrite(str(crop_path), crop)
                    crop_path_str = str(crop_path)
                except Exception:
                    pass

            # OCR processing
            with self._ocr_lock:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray, config=f'--oem 3 --psm {OCR_PSM_PRIMARY} -l {OCR_LANG}')
                text = clean_ocr_text(text)

            return crop_path_str, text if text else ""

        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return "", ""

    def summarize_text(self, text: str) -> str:
        """Text summarization"""
        try:
            if not text or len(text.strip()) < 50:
                return text if text else ""
            
            cleaned_text = clean_ocr_text(text)
            
            if not cleaned_text or len(cleaned_text.strip()) < 50:
                return cleaned_text if cleaned_text else ""
            
            if self.summarizer:
                try:
                    with self._summ_lock:
                        summary = self.summarizer(
                            cleaned_text[:MAX_INPUT_CHARS_FOR_SUMMARY],
                            max_length=MAX_SUMMARY_LENGTH,
                            min_length=30,
                            do_sample=False
                        )
                    return summary[0]['summary_text']
                except Exception as e:
                    self.logger.warning(f"Summarization error: {e}")
            
            # Fallback summary
            sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
            if len(sentences) >= 2:
                return '. '.join(sentences[:2]) + '.'
            else:
                return cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text
                
        except Exception as e:
            self.logger.error(f"Summarization error: {e}")
            return text[:200] + "..." if text and len(text) > 200 else (text if text else "")
