Newspaper Education Extractor
================================

Detects article regions in newspaper PDFs, OCRs them, filters for education-related content, and summarizes results. Includes a CLI and a Streamlit app.

Features
- PDF → images (PyMuPDF)
- Article detection (YOLOv8, local weights)
- OCR (Tesseract)
- Keyword filtering (education domain)
- Summarization (local Transformers pipeline)
- JSON output + optional crops
- Streamlit UI for upload and review

Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
brew install tesseract # macOS; use your OS package manager otherwise
```

Place YOLO weights at `models/best.pt`.

CLI Usage
```bash
python main.py input/newspaper.pdf \
  --save-crops \
  --conf-threshold 0.72 \
  --min-keywords 3 \
  --workers 6
```
- Results: `output/results/<pdf_name>_education_articles.json`
- Images: `output/images/`
- Crops (with `--save-crops`): `output/crops/`

Streamlit App
```bash
streamlit run app.py
```
Upload a PDF, adjust thresholds, and view interactive summaries. Download the JSON from the UI.

Config
See `config.py` for defaults:
- Detection thresholds, OCR settings, summarization model, concurrency, etc.
- All can be overridden via CLI and the app UI (LLM stays local as configured).

Notes
- Requires Tesseract installed and on PATH.
- GPU optional; Transformers will use CUDA if available.

License
MIT


