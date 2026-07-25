#!/bin/bash
set -e

echo '=========================================='
echo '  Book & Paper Recommender - Startup'
echo '=========================================='

# Show available memory
echo "Available memory:"
free -m || cat /proc/meminfo | grep MemAvailable

if [ ! -f "/app/models_store/faiss_tfidf.index" ]; then
    echo 'Building FAISS TF-IDF index in batches...'
    python -c "
import faiss
import scipy.sparse as sp
import numpy as np
import gc

print('Loading TF-IDF matrix...')
mat = sp.load_npz('/app/models_store/tfidf_matrix.npz')
n_docs, dim = mat.shape
print(f'Matrix: {n_docs} docs x {dim} features')

index = faiss.IndexFlatIP(dim)
batch_size = 100

for start in range(0, n_docs, batch_size):
    end = min(start + batch_size, n_docs)
    batch = mat[start:end].toarray().astype('float32')
    faiss.normalize_L2(batch)
    index.add(batch)
    del batch
    gc.collect()
    if start % 1000 == 0:
        print(f'  Progress: {end}/{n_docs}')

faiss.write_index(index, '/app/models_store/faiss_tfidf.index')
del index
gc.collect()
print(f'Done')
"
    echo 'FAISS index ready!'
else
    echo 'FAISS index exists, skipping.'
fi

echo 'Starting Streamlit...'
exec streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false