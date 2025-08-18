#!/usr/bin/env python3
"""
Newspaper Education Article Extractor - Main Entry Point
"""

import os
import sys
import argparse
from pathlib import Path

# Set tokenizers parallelism to false to avoid deadlocks when forking
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import NewspaperEducationExtractor


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Extract and summarize education articles from newspaper PDFs (Optimized)'
    )
    parser.add_argument(
        'pdf_path', 
        help='Path to the newspaper PDF file'
    )
    parser.add_argument(
        '--min-keywords', type=int, default=None,
        help='Minimum number of education keywords required to keep an article (overrides config)'
    )
    parser.add_argument(
        '--conf-threshold', type=float, default=None,
        help='YOLO confidence threshold (overrides config)'
    )
    parser.add_argument(
        '--summarizer', type=str, default=None,
        help='Summarization model name (overrides config)'
    )
    parser.add_argument(
        '--workers', type=int, default=None,
        help='Number of worker threads for parallel processing (overrides config)'
    )
    parser.add_argument(
        '--save-crops', action='store_true',
        help='Persist cropped article images to disk (default: False)'
    )
    parser.add_argument(
        '--fast-mode', action='store_true',
        help='Enable fast processing mode (sacrifices some accuracy for speed)'
    )
    
    args = parser.parse_args()
    
    # Validate PDF path
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    try:
        # Initialize extractor
        print("Initializing Optimized Newspaper Education Extractor...")
        extractor = NewspaperEducationExtractor(
            min_keyword_matches=args.min_keywords,
            confidence_threshold=args.conf_threshold,
            summarization_model=args.summarizer,
            num_workers=args.workers,
            save_crops=args.save_crops,
        )
        
        # Process the newspaper
        print(f"Processing: {pdf_path}")
        if args.fast_mode:
            print("Fast mode enabled - optimizing for speed")
        
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
