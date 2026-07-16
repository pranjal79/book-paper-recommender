FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .

RUN pip install --upgrade pip

# Install packages in groups so failures are easier to debug
RUN pip install numpy pandas scikit-learn pyyaml tqdm requests

RUN pip install nltk spacy beautifulsoup4 lxml

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install sentence-transformers faiss-cpu scipy

RUN pip install mlflow python-dotenv gitpython streamlit

RUN pip install pytest pytest-cov flake8

# ── Download NLP assets ───────────────────────────────────────────────────────
RUN python -c "import nltk; print('nltk version:', nltk.__version__)"

RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True); \
print('NLTK assets downloaded')"

RUN python -m spacy download en_core_web_sm

# Pre-download Sentence Transformer model
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('all-MiniLM-L6-v2'); \
print('Sentence Transformer model ready')"

# ── Copy application code ─────────────────────────────────────────────────────
COPY setup.py .
COPY params.yaml .
COPY src/ ./src/
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

RUN pip install -e .

# ── Copy model artifacts ──────────────────────────────────────────────────────
COPY models_store/ ./models_store/
COPY data/processed/combined_processed.csv ./data/processed/combined_processed.csv
COPY data/raw/books_raw.csv ./data/raw/books_raw.csv
COPY data/raw/papers_raw.csv ./data/raw/papers_raw.csv

# ── Security: non-root user ───────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]