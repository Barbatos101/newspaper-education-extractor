FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models output/images output/crops output/results

# Expose Cloud Run port
EXPOSE 8501

# Set environment variables for optimization
ENV NUM_WORKERS=1
ENV REDUCED_DPI=150
ENV PORT=8501

# Run Streamlit with Cloud Run configuration
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
