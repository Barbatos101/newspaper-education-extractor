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
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from config import *

class NewspaperEducationExtractor:
    def __init__(
        self,
        min_keyword_matches: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        summarization_model: Optional[str] = None,
        num_workers: Optional[int] = None,
        save_crops: bool = False,
    ):
        """Initialize the extractor with local models and runtime settings"""
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Runtime settings (with config fallbacks)
        self.keyword_min_match = (
            min_keyword_matches if min_keyword_matches is not None else KEYWORD_MIN_MATCH
        )
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else CONFIDENCE_THRESHOLD
        )
        self.num_workers = num_workers if num_workers is not None else NUM_WORKERS
        self.save_crops = save_crops

        # Load YOLOv8 model
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        self.yolo_model = YOLO(str(MODEL_PATH))
        self.logger.info(f"Loaded YOLOv8 model from: {MODEL_PATH}")
        
        # Initialize local summarization model
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
        except:
            self.logger.warning("Summarization model failed to load, using simple truncation")
            self.summarizer = None
        
        self.logger.info("Extractor initialized successfully")

    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[str]:
        """Convert PDF pages to images"""
        self.logger.info(f"Converting PDF: {pdf_path}")
        
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
            
            self.logger.info(f"Converted page {page_num + 1}/{pdf_document.page_count}")
        
        pdf_document.close()
        return image_paths

    def detect_articles(self, image_path: str) -> List[Dict]:
        """Detect articles using trained YOLOv8 model"""
        results = self.yolo_model.predict(
            source=image_path,
            conf=self.confidence_threshold,
            save=False,
            verbose=False
        )
        
        articles = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                x1, y1, x2, y2 = map(int, box)
                articles.append({
                    'article_id': i + 1,
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'image_path': image_path
                })
        
        self.logger.info(
            f"Detected {len(articles)} articles with confidence > {self.confidence_threshold}"
        )
        return articles

    def _pad_bbox(self, bbox: List[int], image_shape: Tuple[int, int, int]) -> List[int]:
        """Expand bbox by a percentage while clamping to image bounds"""
        x1, y1, x2, y2 = bbox
        height, width = image_shape[:2]
        pad_x = int((x2 - x1) * BBOX_PADDING_PCT)
        pad_y = int((y2 - y1) * BBOX_PADDING_PCT)
        x1p = max(0, x1 - pad_x)
        y1p = max(0, y1 - pad_y)
        x2p = min(width - 1, x2 + pad_x)
        y2p = min(height - 1, y2 + pad_y)
        return [x1p, y1p, x2p, y2p]

    def _preprocess_for_ocr(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Simple OCR preprocessing: grayscale, denoise, binarize, slight morphological open"""
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        # Adaptive threshold can help with uneven lighting; fallback to Otsu if needed
        thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
        kernel = np.ones((1, 1), np.uint8)
        opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
        return opened

    def extract_article_crop_and_text(self, article_data: Dict) -> Tuple[str, str]:
        """Extract article crop and perform OCR"""
        # Load image and pad bbox
        img = cv2.imread(article_data['image_path'])
        x1, y1, x2, y2 = self._pad_bbox(article_data['bbox'], img.shape)
        crop = img[y1:y2, x1:x2]

        crop_path_str = ""
        if self.save_crops:
            image_name = Path(article_data['image_path']).stem
            crop_filename = (
                f"{image_name}_article_{article_data['article_id']}_conf{article_data['confidence']:.2f}.jpg"
            )
            crop_path = OUTPUT_DIR / "crops" / crop_filename
            cv2.imwrite(str(crop_path), crop)
            crop_path_str = str(crop_path)
        
        # OCR extraction with preprocessing and fallback PSM
        try:
            pre = self._preprocess_for_ocr(crop)
            ocr_config = f"--oem 3 --psm {OCR_PSM_PRIMARY} -l {OCR_LANG}"
            text = pytesseract.image_to_string(pre, config=ocr_config)
            if len(text.strip()) < 30:
                # fallback to a different PSM
                ocr_config_fb = f"--oem 3 --psm {OCR_PSM_FALLBACK} -l {OCR_LANG}"
                text = pytesseract.image_to_string(pre, config=ocr_config_fb)
            text = re.sub(r'\s+', ' ', text).strip()
            return crop_path_str, text
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return crop_path_str, ""

    def contains_education_keywords(self, text: str, min_keywords: int = 2) -> Tuple[bool, List[str]]:
        """Check if text contains education-related keywords"""
        text_lower = text.lower()
        found_keywords = [kw for kw in EDUCATION_KEYWORDS if kw in text_lower]
        is_education = len(found_keywords) >= min_keywords
        return is_education, found_keywords

    def summarize_text(self, text: str) -> str:
        """Summarize text using local model"""
        if not text or len(text.strip()) < 50:
            return text
        
        if self.summarizer:
            try:
                # Summarize a limited input length for performance
                summary = self.summarizer(
                    text[:MAX_INPUT_CHARS_FOR_SUMMARY],
                    max_length=MAX_SUMMARY_LENGTH,
                    min_length=30,
                    do_sample=False
                )
                return summary[0]['summary_text']
            except Exception as e:
                self.logger.error(f"Summarization error: {e}")
        
        # Fallback: simple truncation
        sentences = text.split('. ')
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        return text[:200] + "..." if len(text) > 200 else text

    def _process_single_article(self, article: Dict, page_num: int) -> Optional[Dict]:
        """Process one detected article and return education article dict or None"""
        crop_path, text = self.extract_article_crop_and_text(article)
        if len(text.strip()) < 50:
            return None
        is_education, keywords = self.contains_education_keywords(text, self.keyword_min_match)
        if not is_education:
            return None
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
        """Complete processing pipeline"""
        self.logger.info(f"Processing newspaper: {pdf_path}")
        
        # Step 1: Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)
        
        # Initialize results
        education_articles = []
        stats = {
            'total_pages': len(image_paths),
            'total_articles_detected': 0,
            'education_articles_found': 0
        }
        
        # Step 2: Process each page
        for page_num, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing page {page_num}/{len(image_paths)}")
            
            # Detect articles
            articles = self.detect_articles(image_path)
            stats['total_articles_detected'] += len(articles)
            
            # Process articles in parallel
            if articles:
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._process_single_article, article, page_num): idx
                        for idx, article in enumerate(articles)
                    }
                    for future in as_completed(future_to_idx):
                        result = future.result()
                        if result:
                            education_articles.append(result)
                            stats['education_articles_found'] += 1
                            self.logger.info(
                                f"Found education article: Page {result['page']}, "
                                f"Article {result['article_id']}, "
                                f"Keywords: {result['keywords_found'][:3]}"
                            )
        
        # Step 3: Save results
        results = {
            'pdf_path': pdf_path,
            'processing_stats': stats,
            'education_articles': education_articles,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Save to JSON
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
        print("NEWSPAPER EDUCATION ARTICLE EXTRACTION RESULTS")
        print("="*70)
        print(f"PDF: {Path(results['pdf_path']).name}")
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
                print(f"   Summary: {article['summary']}")
        else:
            print("\nNo education-related articles found.")
        
        print("\n" + "="*70)
