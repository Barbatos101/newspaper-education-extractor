import tempfile
from pathlib import Path
import json
import os

import streamlit as st

# Set environment variables before importing other modules
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from extractor import NewspaperEducationExtractor
from config import CONFIDENCE_THRESHOLD, KEYWORD_MIN_MATCH, NUM_WORKERS

st.set_page_config(page_title="Newspaper Education Extractor (Semantic)", layout="wide")
st.title("Newspaper Education Extractor with Semantic Features")
st.caption("Upload a newspaper PDF to detect, OCR, and summarize education-related articles using semantic analysis and sshleifer/distilbart-cnn-12-6 summarization.")

def main():
    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("YOLO confidence threshold", 0.3, 0.95, value=float(CONFIDENCE_THRESHOLD), step=0.01)
        min_keywords = st.slider("Min education keywords", 1, 5, value=int(KEYWORD_MIN_MATCH), step=1)
        workers = st.slider("Workers", 1, 8, value=int(NUM_WORKERS), step=1)
        save_crops = st.checkbox("Save cropped images", value=False)
        
        st.info("🧠 Semantic filtering enabled for better accuracy")
        st.info("📝 Using sshleifer/distilbart-cnn-12-6 for summarization")
        
        run_button = st.button("Run Extraction", type="primary")

    uploaded_pdf = st.file_uploader("Upload newspaper PDF", type=["pdf"]) 

    if run_button:
        if not uploaded_pdf:
            st.warning("Please upload a PDF first.")
            st.stop()

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

        with st.spinner("Processing PDF with semantic analysis... This may take a few minutes."):
            results = extractor.process_newspaper(tmp_path)

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
                    meta_cols[1].write(f"**Text length:** {article.get('text_length', 0)} chars")
                    meta_cols[2].write(f"**BBox:** {article.get('bbox', [])}")
                    
                    # Show crop if available
                    if article.get("crop_path") and Path(article["crop_path"]).exists():
                        st.image(str(article["crop_path"]), caption="Article Crop", use_container_width=True)
                    
                    # Summary (from sshleifer model)
                    st.markdown("**AI Summary** (sshleifer/distilbart-cnn-12-6)")
                    st.write(article.get("summary", ""))
                    
                    # Full text
                    with st.expander("View full OCR text"):
                        st.text_area("Full text", article.get("full_text", ""), height=200, key=f"text_{i}")
        else:
            st.info("No education-related articles found. The semantic filter may have filtered out irrelevant content.")

        # Download results
        st.subheader("Download Results")
        json_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Download JSON Results", 
            data=json_bytes, 
            file_name=f"education_articles_semantic.json", 
            mime="application/json"
        )
        
        # Performance info
        with st.expander("ℹ️ Processing Details"):
            st.write(f"**Semantic Analysis:** {'Enabled' if results.get('semantic_enabled', False) else 'Disabled'}")
            st.write(f"**Summarization Model:** {results.get('summarization_model', 'N/A')}")
            st.write(f"**Processing Timestamp:** {results.get('timestamp', 'N/A')}")
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass

if __name__ == "__main__":
    main()
