FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install numpy pandas scikit-learn pyyaml tqdm requests
RUN pip install nltk spacy beautifulsoup4 lxml
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install sentence-transformers faiss-cpu scipy
RUN pip install mlflow python-dotenv gitpython streamlit
RUN pip install pytest pytest-cov flake8

RUN python -c "import nltk; nltk.download('stopwords',quiet=True); nltk.download('wordnet',quiet=True); nltk.download('omw-1.4',quiet=True)"
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY setup.py .
COPY params.yaml .
COPY src/ ./src/
COPY app/ ./app/
RUN sed -i 's/\r//' ./app/startup.sh
COPY .streamlit/ ./.streamlit/
RUN pip install -e .

# Copy model artifacts (faiss_tfidf.index rebuilt at startup)
COPY models_store/tfidf_vectorizer.pkl    ./models_store/tfidf_vectorizer.pkl
COPY models_store/tfidf_matrix.npz        ./models_store/tfidf_matrix.npz
COPY models_store/sentence_embeddings.npy ./models_store/sentence_embeddings.npy
COPY models_store/faiss_semantic.index    ./models_store/faiss_semantic.index
COPY models_store/metadata.csv            ./models_store/metadata.csv

# Copy data CSVs
COPY data/processed/combined_processed.csv ./data/processed/combined_processed.csv
COPY data/raw/books_raw.csv               ./data/raw/books_raw.csv
COPY data/raw/papers_raw.csv             ./data/raw/papers_raw.csv

RUN chmod +x ./app/startup.sh

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

CMD ["./app/startup.sh"]