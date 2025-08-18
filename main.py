#!/usr/bin/env python3
"""
Newspaper Education Article Extractor - Semantic Version
"""

import os
import sys
import argparse
from pathlib import Path

# Set tokenizers parallelism to false
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import NewspaperEducationExtractor


def main():
    parser = argparse.ArgumentParser(
        description='Extract and summarize education articles from newspaper PDFs using semantic analysis'
    )
    parser.add_argument('pdf_path', help='Path to the newspaper PDF file')
    parser.add_argument('--min-keywords', type=int, default=None, help='Minimum education keywords required')
    parser.add_argument('--conf-threshold', type=float, default=None, help='YOLO confidence threshold')
    parser.add_argument('--summarizer', type=str, default=None, help='Summarization model name')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker threads')
    parser.add_argument('--save-crops', action='store_true', help='Save cropped article images')
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        print("Initializing Semantic Newspaper Education Extractor...")
        print("Features: Semantic filtering + sshleifer/distilbart-cnn-12-6 summarization")
        
        extractor = NewspaperEducationExtractor(
            min_keyword_matches=args.min_keywords,
            confidence_threshold=args.conf_threshold,
            summarization_model=args.summarizer,
            num_workers=args.workers,
            save_crops=args.save_crops,
        )
        
        print(f"Processing: {pdf_path}")
        results = extractor.process_newspaper(str(pdf_path))
        
        # Display results
        extractor.print_summary(results)
        
        print(f"\nDetailed results saved to: output/results/")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
