#!/bin/bash
set -e

echo '=========================================='
echo '  Book & Paper Recommender - Startup'
echo '=========================================='

if [ ! -f "/app/models_store/faiss_tfidf.index" ]; then
    echo 'Building FAISS TF-IDF index in batches (memory efficient)...'
    python -c "
import faiss
import scipy.sparse as sp
import numpy as np

print('Loading TF-IDF matrix...')
mat = sp.load_npz('/app/models_store/tfidf_matrix.npz')
n_docs, dim = mat.shape
print(f'Matrix: {n_docs} docs x {dim} features')

index = faiss.IndexFlatIP(dim)
batch_size = 200

for start in range(0, n_docs, batch_size):
    end = min(start + batch_size, n_docs)
    batch = mat[start:end].toarray().astype('float32')
    faiss.normalize_L2(batch)
    index.add(batch)
    if start % 2000 == 0:
        print(f'  Progress: {end}/{n_docs}')

faiss.write_index(index, '/app/models_store/faiss_tfidf.index')
print(f'Done: {index.ntotal} vectors indexed')
"
    echo 'FAISS TF-IDF index ready!'
else
    echo 'FAISS TF-IDF index exists, skipping rebuild.'
fi

echo 'Starting Streamlit...'
exec streamlit run app/streamlit_app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false