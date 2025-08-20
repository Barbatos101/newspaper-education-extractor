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

def main():
    # Initialize session state
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "results" not in st.session_state:
        st.session_state.results = None
    if "uploaded_file_name" not in st.session_state:
        st.session_state.uploaded_file_name = None

    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("YOLO confidence threshold", 0.3, 0.95, value=float(CONFIDENCE_THRESHOLD), step=0.01)
        min_keywords = st.slider("Min education keywords", 1, 5, value=int(KEYWORD_MIN_MATCH), step=1)
        workers = st.slider("Workers", 1, 8, value=int(NUM_WORKERS), step=1)
        save_crops = st.checkbox("Save cropped images", value=False)
        
        st.info("🧠 Semantic filtering enabled")
        st.info("📝 Using sshleifer/distilbart-cnn-12-6")

    # File uploader with size limit
    uploaded_pdf = st.file_uploader("Upload newspaper PDF", type=["pdf"])

    # File size validation
    if uploaded_pdf is not None:
        file_size_mb = uploaded_pdf.size / (1024 * 1024)
        max_size_mb = 50  # 50MB limit for Cloud Run compatibility
        
        if file_size_mb > max_size_mb:
            st.error(f"File too large ({file_size_mb:.1f}MB). Please upload a file smaller than {max_size_mb}MB.")
            st.info("💡 Tip: Try compressing your PDF or splitting large files into smaller ones.")
            return  # Early return instead of st.stop()
        else:
            st.success(f"File uploaded successfully ({file_size_mb:.1f}MB)")
            st.session_state.uploaded_file_name = uploaded_pdf.name

    # Run extraction button with session state
    run_extraction = st.button("Run Extraction", type="primary", disabled=st.session_state.processing)

    if run_extraction and uploaded_pdf is not None:
        st.session_state.processing = True
        st.rerun()  # Trigger rerun to show processing state

    # Processing logic
    if st.session_state.processing and uploaded_pdf is not None:
        # Write to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Create extractor
        extractor = NewspaperEducationExtractor(
            min_keyword_matches=min_keywords,
            confidence_threshold=conf_threshold,
            num_workers=workers,
            save_crops=save_crops,
        )

        # Processing with progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Initializing extraction...")
            progress_bar.progress(10)
            
            status_text.text("Processing PDF... This may take a few minutes.")
            progress_bar.progress(30)
            
            results = extractor.process_newspaper(tmp_path)
            
            progress_bar.progress(100)
            status_text.text("Processing complete!")
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.processing = False
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            st.rerun()  # Trigger rerun to show results
            
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.info("This might be due to file size or complexity. Try with a smaller PDF.")
            st.session_state.processing = False
            try:
                os.unlink(tmp_path)
            except:
                pass

    # Display results if available
    if st.session_state.results is not None:
        results = st.session_state.results
        
        # Display summary
        stats = results.get("processing_stats", {})
        st.subheader("Processing Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Pages processed", stats.get("total_pages", 0))
        col2.metric("Articles detected", stats.get("total_articles_detected", 0))
        col3.metric("Education articles", stats.get("education_articles_found", 0))
        col4.metric("Semantic enabled", "✅" if results.get("semantic_enabled", False) else "❌")

        # Show education articles
        articles = results.get("education_articles", [])
        if articles:
            st.subheader(f"Education Articles ({len(articles)} found)")
            
            # Filtering options
            col1, col2 = st.columns(2)
            with col1:
                keyword_filter = st.selectbox(
                    "Filter by keyword:",
                    ["All"] + sorted(set(kw for article in articles for kw in article.get('keywords_found', []))),
                    index=0
                )
            with col2:
                min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
            
            # Apply filters
            filtered_articles = articles
            if keyword_filter != "All":
                filtered_articles = [a for a in articles if keyword_filter in a.get('keywords_found', [])]
            if min_confidence > 0:
                filtered_articles = [a for a in articles if a.get('confidence', 0) >= min_confidence]
            
            for i, article in enumerate(filtered_articles, 1):
                with st.expander(f"{i}. Page {article['page']} • Article {article['article_id']} • conf={article['confidence']:.2f}"):
                    # Metadata
                    meta_cols = st.columns(3)
                    meta_cols[0].write(f"**Keywords:** {', '.join(article.get('keywords_found', [])[:6])}")
                    meta_cols[13].write(f"**Text length:** {article.get('text_length', 0)} chars")
                    meta_cols.write(f"**BBox:** {article.get('bbox', [])}")
                    
                    # Show crop if available
                    if article.get("crop_path") and Path(article["crop_path"]).exists():
                        st.image(str(article["crop_path"]), caption="Article Crop", use_container_width=True)
                    
                    # Summary
                    st.markdown("**AI Summary**")
                    st.write(article.get("summary", ""))
                    
                    # Full text
                    with st.expander("View full OCR text"):
                        st.text_area("Full text", article.get("full_text", ""), height=200, key=f"text_{i}")
        else:
            st.info("No education-related articles found.")

        # Download results
        st.subheader("Download Results")
        json_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Download JSON Results", 
            data=json_bytes, 
            file_name=f"education_articles_{st.session_state.uploaded_file_name}.json", 
            mime="application/json"
        )
        
        # Clear results button
        if st.button("Process Another PDF"):
            st.session_state.results = None
            st.session_state.processing = False
            st.session_state.uploaded_file_name = None
            st.rerun()

if __name__ == "__main__":
    main()
