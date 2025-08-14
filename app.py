import tempfile
from pathlib import Path
import json

import streamlit as st

from extractor import NewspaperEducationExtractor
from config import CONFIDENCE_THRESHOLD, KEYWORD_MIN_MATCH, NUM_WORKERS


st.set_page_config(page_title="Newspaper Education Extractor", layout="wide")
st.title("Newspaper Education Extractor")
st.caption("Upload a newspaper PDF to detect, OCR, and summarize education-related articles.")


def main():
    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("YOLO confidence threshold", 0.3, 0.95, value=float(CONFIDENCE_THRESHOLD), step=0.01)
        min_keywords = st.slider("Min education keywords", 1, 5, value=int(KEYWORD_MIN_MATCH), step=1)
        workers = st.slider("Workers", 1, 12, value=int(NUM_WORKERS), step=1)
        save_crops = st.checkbox("Save cropped images", value=False)
        run_button = st.button("Run Extraction", type="primary")

    uploaded_pdf = st.file_uploader("Upload newspaper PDF", type=["pdf"]) 

    if run_button:
        if not uploaded_pdf:
            st.warning("Please upload a PDF first.")
            st.stop()

        # Write to a temp file on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name

        # Create extractor once with UI overrides
        extractor = NewspaperEducationExtractor(
            min_keyword_matches=min_keywords,
            confidence_threshold=conf_threshold,
            num_workers=workers,
            save_crops=save_crops,
        )

        with st.spinner("Processing PDF... This may take a few minutes."):
            results = extractor.process_newspaper(tmp_path)

        # Display summary
        stats = results.get("processing_stats", {})
        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Pages processed", stats.get("total_pages", 0))
        col2.metric("Articles detected", stats.get("total_articles_detected", 0))
        col3.metric("Education articles", stats.get("education_articles_found", 0))

        # Show table of education articles
        articles = results.get("education_articles", [])
        if articles:
            st.subheader("Education Articles")
            for i, article in enumerate(articles, 1):
                with st.expander(f"{i}. Page {article['page']} • Article {article['article_id']} • conf={article['confidence']:.2f}"):
                    meta_cols = st.columns(3)
                    meta_cols[0].write(f"Keywords: {', '.join(article.get('keywords_found', [])[:5])}")
                    meta_cols[1].write(f"Text length: {article.get('text_length', 0)}")
                    meta_cols[2].write(f"BBox: {article.get('bbox', [])}")
                    if article.get("crop_path") and Path(article["crop_path"]).exists():
                        st.image(str(article["crop_path"]), caption="Crop", use_container_width=True)
                    st.markdown("**Summary**")
                    st.write(article.get("summary", ""))
                    with st.expander("Full OCR text"):
                        st.write(article.get("full_text", ""))

        # Offer raw JSON download
        st.subheader("Download Results")
        json_bytes = json.dumps(results, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button("Download JSON", data=json_bytes, file_name="education_articles.json", mime="application/json")


if __name__ == "__main__":
    main()


