import os
import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
import json
import re
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import threading
import platform

from config import *

# Set environment variables for optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"


class SemanticEducationFilter:
    """Semantic education filtering using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.logger = logging.getLogger(__name__)
        
        try:
            self.sentence_model = SentenceTransformer(model_name)
            self.semantic_available = True
            self.logger.info(f"Loaded semantic filter model: {model_name}")
        except Exception as e:
            self.logger.error(f"Semantic model loading failed: {e}")
            self.sentence_model = None
            self.semantic_available = False

        # Comprehensive education keywords
        self.education_keywords = EDUCATION_KEYWORDS
        
        # Create embeddings for education keywords
        if self.semantic_available:
            try:
                self.keyword_embeddings = self.sentence_model.encode(self.education_keywords)
                self.logger.info("Created education keyword embeddings")
            except Exception as e:
                self.logger.error(f"Failed to create embeddings: {e}")
                self.semantic_available = False
        
        # Education context patterns
        self.education_contexts = [
            "school education", "student performance", "teacher training",
            "educational system", "learning outcomes", "curriculum development",
            "academic achievement", "classroom instruction", "school administration",
            "educational policy", "student assessment", "teacher evaluation"
        ]
        
        if self.semantic_available:
            try:
                self.context_embeddings = self.sentence_model.encode(self.education_contexts)
            except Exception as e:
                self.logger.error(f"Failed to create context embeddings: {e}")

    def is_education_article(self, text: str, min_keywords: int = 2, semantic_threshold: float = SEMANTIC_THRESHOLD) -> Tuple[bool, List[str]]:
        """Determine if text is education-related using semantic analysis"""
        
        if not text or len(text.strip()) < 30:
            return False, []
        
        text_lower = text.lower()
        
        # Step 1: Traditional keyword matching
        found_keywords = []
        for keyword in self.education_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                found_keywords.append(keyword)
        
        # Step 2: Semantic similarity (if available)
        semantic_score = 0.0
        context_score = 0.0
        
        if self.semantic_available and self.sentence_model:
            try:
                # Encode the input text
                text_embedding = self.sentence_model.encode([text])
                
                # Calculate similarity with education keywords
                keyword_similarities = cosine_similarity(text_embedding, self.keyword_embeddings)
                semantic_score = np.max(keyword_similarities[0])
                
                # Calculate similarity with education contexts
                context_similarities = cosine_similarity(text_embedding, self.context_embeddings)
                context_score = np.max(context_similarities[0])
                
            except Exception as e:
                self.logger.warning(f"Semantic analysis failed: {e}")
        
        # Step 3: Combined decision making
        keyword_score = len(found_keywords)
        
        # Multiple criteria for classification
        criteria_met = 0
        
        # Criterion 1: Sufficient keywords
        if keyword_score >= min_keywords:
            criteria_met += 1
        
        # Criterion 2: High semantic similarity
        if semantic_score >= semantic_threshold:
            criteria_met += 1
        
        # Criterion 3: Strong context match
        if context_score >= semantic_threshold:
            criteria_met += 1
        
        # Criterion 4: Combined score threshold
        combined_score = (keyword_score * 0.4) + (semantic_score * 0.35) + (context_score * 0.25)
        if combined_score >= 1.5:
            criteria_met += 1
        
        # Decision: Need at least 1 criterion if semantic available, otherwise just keywords
        is_education = criteria_met >= 1 if self.semantic_available else keyword_score >= min_keywords
        
        return is_education, found_keywords


class OptimizedArticleDetector:
    """Optimized article detection for speed while maintaining accuracy"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.68):
        self.yolo_model = YOLO(str(model_path))
        self.conf_threshold = confidence_threshold

    def detect_articles(self, image_path: str) -> List[Dict]:
        """Optimized single-scale detection"""
        results = self.yolo_model.predict(
            source=image_path,
            conf=self.conf_threshold,
            imgsz=640,  # Balanced size for speed/accuracy
            verbose=False,
            save=False
        )
        
        detections = self._extract_detections(results)
        return self._layout_filtering(detections, cv2.imread(image_path).shape)

    def _extract_detections(self, results) -> List[Dict]:
        """Extract detections with safety checks"""
        detections = []
        
        try:
            if not results or len(results) == 0:
                return detections
                
            result = results[0]
            if not hasattr(result, 'boxes') or result.boxes is None:
                return detections
                
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)
                
                # Filter by area and aspect ratio
                if area > 4000:  # Reasonable minimum area
                    aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
                    if 0.3 <= aspect_ratio <= 8.0:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'area': area
                        })
                        
        except Exception as e:
            logging.warning(f"Error extracting detections: {e}")
            
        return detections

    def _layout_filtering(self, detections: List[Dict], img_shape: Tuple) -> List[Dict]:
        """Basic layout-aware filtering"""
        height, width = img_shape[:2]
        filtered = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            rel_width = (x2 - x1) / width
            rel_height = (y2 - y1) / height
            
            # Accept reasonable sizes
            if 0.12 <= rel_width <= 0.95 and 0.06 <= rel_height <= 0.9:
                filtered.append(det)
        
        return filtered


class FastOCRProcessor:
    """Optimized OCR processor for speed"""
    
    def __init__(self):
        pass

    def extract_text_fast(self, image_crop: np.ndarray) -> str:
        """Fast OCR with basic preprocessing"""
        
        # Resize if too large for speed
        height, width = image_crop.shape[:2]
        if height > 1200 or width > 1200:
            scale = min(1200/height, 1200/width)
            new_height, new_width = int(height * scale), int(width * scale)
            image_crop = cv2.resize(image_crop, (new_width, new_height))
        
        # Basic preprocessing
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop.copy()
        
        # Simple thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # OCR
        try:
            config = f'--oem 3 --psm {OCR_PSM_PRIMARY} -l {OCR_LANG}'
            text = pytesseract.image_to_string(thresh, config=config)
            
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text).strip()
            return text
            
        except Exception as e:
            logging.warning(f"OCR failed: {e}")
            return ""


class NewspaperEducationExtractor:
    def __init__(
        self,
        min_keyword_matches: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        summarization_model: Optional[str] = None,
        num_workers: Optional[int] = None,
        save_crops: bool = False,
    ):
        """Initialize the semantic-enabled extractor"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Configure OpenCV
        cv2.setNumThreads(0)
        try:
            cv2.ocl.setUseOpenCL(False)
        except:
            pass

        # Runtime settings
        self.keyword_min_match = min_keyword_matches if min_keyword_matches is not None else KEYWORD_MIN_MATCH
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
        self.num_workers = num_workers if num_workers is not None else NUM_WORKERS
        self.save_crops = save_crops
        
        # Threading locks
        self._ocr_lock = threading.Lock()
        self._summ_lock = threading.Lock()

        # Load model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        # Initialize components
        self.detector = OptimizedArticleDetector(MODEL_PATH, self.confidence_threshold)
        self.ocr_processor = FastOCRProcessor()
        self.education_filter = SemanticEducationFilter()
        
        self.logger.info("Loaded optimized article detector")
        self.logger.info("Loaded fast OCR processor")
        self.logger.info("Loaded semantic education filter")
        
        # Initialize summarization with specified model
        self.logger.info("Loading summarization model...")
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

        self.logger.info("Extractor initialized successfully with semantic features")

    def pdf_to_images(self, pdf_path: str, dpi: int = None) -> List[str]:
        """Convert PDF pages to images with optimized DPI"""
        if dpi is None:
            dpi = REDUCED_DPI
            
        self.logger.info(f"Converting PDF: {pdf_path} at {dpi} DPI")
        
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
            
            if page_num % 3 == 0:  # Log every 3 pages
                self.logger.info(f"Converted page {page_num + 1}/{pdf_document.page_count}")
        
        pdf_document.close()
        return image_paths

    def detect_articles(self, image_path: str) -> List[Dict]:
        """Detect articles using optimized detection"""
        articles = self.detector.detect_articles(image_path)
        
        # Add article IDs and image path
        for i, article in enumerate(articles):
            article['article_id'] = i + 1
            article['image_path'] = image_path
        
        return articles

    def _pad_bbox(self, bbox: List[int], image_shape: Tuple[int, int, int]) -> List[int]:
        """Expand bbox by padding percentage"""
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]
        pad_x = int((x2 - x1) * BBOX_PADDING_PCT)
        pad_y = int((y2 - y1) * BBOX_PADDING_PCT)
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(width - 1, x2 + pad_x)
        y2p = min(height - 1, y2 + pad_y)
        return [x1p, y1p, x2p, y2p]

    def extract_article_crop_and_text(self, article_data: Dict) -> Tuple[str, str]:
        """Extract article crop and perform fast OCR"""
        # Load image and crop
        img = cv2.imread(article_data['image_path'])
        x1, y1, x2, y2 = self._pad_bbox(article_data['bbox'], img.shape)
        crop = img[y1:y2, x1:x2]

        crop_path_str = ""
        if self.save_crops:
            image_name = Path(article_data['image_path']).stem
            crop_filename = f"{image_name}_article_{article_data['article_id']}_conf{article_data['confidence']:.2f}.jpg"
            crop_path = OUTPUT_DIR / "crops" / crop_filename
            cv2.imwrite(str(crop_path), crop)
            crop_path_str = str(crop_path)
        
        # Fast OCR
        try:
            with self._ocr_lock:
                text = self.ocr_processor.extract_text_fast(crop)
            return crop_path_str, text
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return crop_path_str, ""

    def contains_education_keywords(self, text: str, min_keywords: int = 2) -> Tuple[bool, List[str]]:
        """Check if text is education-related using semantic filtering"""
        try:
            is_education, keywords = self.education_filter.is_education_article(text, min_keywords)
            return is_education, keywords
        except Exception as e:
            self.logger.error(f"Education filtering error: {e}")
            # Fallback to simple keyword matching
            text_lower = text.lower()
            found_keywords = [kw for kw in EDUCATION_KEYWORDS if kw in text_lower]
            return len(found_keywords) >= min_keywords, found_keywords

    def summarize_text(self, text: str) -> str:
        """Summarize text using specified model"""
        if not text or len(text.strip()) < 50:
            return text
        
        if self.summarizer:
            try:
                with self._summ_lock:
                    summary = self.summarizer(
                        text[:MAX_INPUT_CHARS_FOR_SUMMARY],
                        max_length=MAX_SUMMARY_LENGTH,
                        min_length=30,
                        do_sample=False
                    )
                return summary[0]['summary_text']
            except Exception as e:
                self.logger.warning(f"Summarization error: {e}")
        
        # Fast fallback: extractive summary
        sentences = text.split('. ')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        return text[:180] + "..." if len(text) > 180 else text

    def _process_single_article(self, article: Dict, page_num: int) -> Optional[Dict]:
        """Process one article with semantic filtering"""
        crop_path, text = self.extract_article_crop_and_text(article)
        
        # Quick length check
        if len(text.strip()) < 40:
            return None
        
        # Semantic education filtering
        is_education, keywords = self.contains_education_keywords(text, self.keyword_min_match)
        if not is_education:
            return None
        
        # Generate summary
        summary = self.summarize_text(text)
        
        return {
            'page': page_num,
            'article_id': article['article_id'],
            'confidence': article['confidence'],
            'bbox': article['bbox'],
            'keywords_found': keywords,
            'full_text': text,
            'summary': summary,
            'crop_path': crop_path,
            'text_length': len(text)
        }

    def process_newspaper(self, pdf_path: str) -> Dict:
        """Complete processing pipeline with semantic features"""
        self.logger.info(f"Processing newspaper with semantic features: {pdf_path}")
        
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
            articles = self.detect_articles(image_path)
            stats['total_articles_detected'] += len(articles)
            
            # Process articles with threading
            if articles:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._process_single_article, article, page_num): idx
                        for idx, article in enumerate(articles)
                    }
                    
                    for future in as_completed(future_to_idx):
                        try:
                            result = future.result(timeout=45)  # Timeout for safety
                            if result:
                                education_articles.append(result)
                                stats['education_articles_found'] += 1
                                self.logger.info(
                                    f"Found education article: Page {result['page']}, "
                                    f"Article {result['article_id']}, "
                                    f"Keywords: {result['keywords_found'][:3]}"
                                )
                        except Exception as e:
                            self.logger.warning(f"Article processing error: {e}")
        
        # Save results
        results = {
            'pdf_path': pdf_path,
            'processing_stats': stats,
            'education_articles': education_articles,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'semantic_enabled': self.education_filter.semantic_available,
            'summarization_model': SUMMARIZATION_MODEL
        }
        
        pdf_name = Path(pdf_path).stem
        results_file = OUTPUT_DIR / "results" / f"{pdf_name}_education_articles.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Processing complete! Found {len(education_articles)} education articles")
        self.logger.info(f"Results saved to: {results_file}")
        
        return results

    def print_summary(self, results: Dict):
        """Print processing summary"""
        stats = results['processing_stats']
        articles = results['education_articles']
        
        print("\n" + "="*70)
        print("NEWSPAPER EDUCATION ARTICLE EXTRACTION RESULTS (SEMANTIC)")
        print("="*70)
        print(f"PDF: {Path(results['pdf_path']).name}")
        print(f"Semantic Filtering: {'Enabled' if results.get('semantic_enabled', False) else 'Disabled'}")
        print(f"Summarization Model: {results.get('summarization_model', 'N/A')}")
        print(f"Pages processed: {stats['total_pages']}")
        print(f"Total articles detected: {stats['total_articles_detected']}")
        print(f"Education articles found: {stats['education_articles_found']}")
        
        if articles:
            print(f"\nEducation Articles Summary:")
            print("-" * 70)
            
            for i, article in enumerate(articles, 1):
                print(f"\n{i}. Page {article['page']}, Article {article['article_id']}")
                print(f"   Confidence: {article['confidence']:.3f}")
                print(f"   Keywords: {', '.join(article['keywords_found'][:5])}")
                print(f"   Text length: {article['text_length']} chars")
                print(f"   Summary: {article['summary'][:120]}...")
        else:
            print("\nNo education-related articles found.")
        
        print("\n" + "="*70)
