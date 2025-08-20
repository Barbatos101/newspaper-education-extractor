import tempfile
from pathlib import Path
import json
import os

import streamlit as st

# Set environment variables before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import NewspaperEducationExtractor
from config import CONFIDENCE_THRESHOLD, KEYWORD_MIN_MATCH, NUM_WORKERS

st.set_page_config(page_title="Newspaper Education Extractor", layout="wide")
st.title("Newspaper Education Extractor with Semantic Features")
st.caption("Upload a newspaper PDF to detect, OCR, and summarize education-related articles.")

# Health check for Cloud Run
if st.query_params.get("health") == "check":
    st.write("OK")
    st.stop()

def main():
    # Initialize session state ONCE
    if "results" not in st.session_state:
        st.session_state.results = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False

    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("YOLO confidence threshold", 0.3, 0.95, value=float(CONFIDENCE_THRESHOLD), step=0.01)
        min_keywords = st.slider("Min education keywords", 1, 5, value=int(KEYWORD_MIN_MATCH), step=1)
        workers = st.slider("Workers", 1, 8, value=int(NUM_WORKERS), step=1)
        save_crops = st.checkbox("Save cropped images", value=False)
        
        st.info("🧠 Semantic filtering enabled")
        st.info("📝 Using sshleifer/distilbart-cnn-12-6")

    # File uploader
    uploaded_pdf = st.file_uploader("Upload newspaper PDF", type=["pdf"], key="pdf_uploader")

    # File size validation
    if uploaded_pdf is not None:
        file_size_mb = uploaded_pdf.size / (1024 * 1024)
        max_size_mb = 50  # 50MB limit for Cloud Run compatibility
        
        if file_size_mb > max_size_mb:
            st.error(f"File too large ({file_size_mb:.1f}MB). Please upload a file smaller than {max_size_mb}MB.")
            st.info("💡 Tip: Try compressing your PDF or splitting large files into smaller ones.")
            return
        else:
            st.success(f"File uploaded successfully ({file_size_mb:.1f}MB)")
            st.session_state.uploaded_file_name = uploaded_pdf.name

    # Run extraction button - SIMPLIFIED LOGIC
    run_extraction = st.button("Run Extraction", type="primary", key="extract_btn")

    # Process when button is clicked - NO ST.RERUN CALLS
    if run_extraction and uploaded_pdf is not None:
        # Clear previous results
        st.session_state.results = None
        st.session_state.processing_complete = False
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Create extractor
        with st.spinner("Creating extractor..."):
            extractor = NewspaperEducationExtractor(
                min_keyword_matches=min_keywords,
                confidence_threshold=conf_threshold,
                num_workers=workers,
                save_crops=save_crops,
            )

        # Processing with progress
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Starting PDF processing...")
                progress_bar.progress(20)
                
                status_text.text("Processing PDF with AI models... Please wait.")
                progress_bar.progress(50)
                
                # Process the PDF
                results = extractor.process_newspaper(tmp_path)
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.processing_complete = True
                
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"Processing failed: {str(e)}")
                st.info("This might be due to file size or complexity. Try with a smaller PDF.")
                progress_bar.empty()
                status_text.empty()
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                return

    # Display results if available - OUTSIDE THE BUTTON LOGIC
    if st.session_state.results is not None and st.session_state.processing_complete:
        results = st.session_state.results
        
        # Display summary
        stats = results.get("processing_stats", {})
        st.subheader("Processing Summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Pages processed", stats.get("total_pages", 0))
        summary_cols.metric("Articles detected", stats.get("total_articles_detected", 0))
        summary_cols[2].metric("Education articles", stats.get("education_articles_found", 0))
        summary_cols[3].metric("Semantic enabled", "✅" if results.get("semantic_enabled", False) else "❌")

        # Show education articles
        articles = results.get("education_articles", [])
        if articles:
            st.subheader(f"Education Articles ({len(articles)} found)")
            
            # Filtering options
            filter_col1, filter_col2 = st.columns(2)
            with filter_col1:
                keyword_filter = st.selectbox(
                    "Filter by keyword:",
                    ["All"] + sorted(set(kw for article in articles for kw in article.get('keywords_found', []))),
                    index=0,
                    key="keyword_filter"
                )
            with filter_col2:
                min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="conf_filter")
            
            # Apply filters
            filtered_articles = articles
            if keyword_filter != "All":
                filtered_articles = [a for a in articles if keyword_filter in a.get('keywords_found', [])]
            if min_confidence > 0:
                filtered_articles = [a for a in articles if a.get('confidence', 0) >= min_confidence]
            
            # Display articles
            for i, article in enumerate(filtered_articles, 1):
                with st.expander(f"{i}. Page {article['page']} • Article {article['article_id']} • conf={article['confidence']:.2f}"):
                    # Metadata with FIXED column indexing
                    meta_cols = st.columns(3)
                    meta_cols[0].write(f"**Keywords:** {', '.join(article.get('keywords_found', [])[:6])}")
                    meta_cols[1].write(f"**Text length:** {article.get('text_length', 0)} chars")
                    meta_cols[2].write(f"**BBox:** {article.get('bbox', [])}")
                    
                    # Show crop if available
                    if article.get("crop_path") and Path(article["crop_path"]).exists():
                        st.image(str(article["crop_path"]), caption="Article Crop", use_container_width=True)
                    
                    # Summary
                    st.markdown("**AI Summary**")
                    summary_text = article.get("summary", "No summary available")
                    st.write(summary_text)
                    
                    # Full text
                    with st.expander("View full OCR text"):
                        full_text = article.get("full_text", "No text extracted")
                        st.text_area("Full text", full_text, height=200, key=f"text_{article['page']}_{article['article_id']}")
        else:
            st.info("No education-related articles found. Try adjusting the confidence threshold or upload a different PDF.")

        # Download results
        st.subheader("Download Results")
        json_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        filename = f"education_articles_{st.session_state.uploaded_file_name or 'results'}.json"
        st.download_button(
            "Download JSON Results", 
            data=json_bytes, 
            file_name=filename, 
            mime="application/json"
        )
        
        # Reset button
        if st.button("Process Another PDF", key="reset_btn"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]

if __name__ == "__main__":
    main()
