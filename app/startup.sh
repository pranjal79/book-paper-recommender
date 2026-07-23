#!/bin/bash
set -e

echo '=========================================='
echo '  Book & Paper Recommender - Startup'
echo '=========================================='

if [ ! -f "/app/models_store/faiss_tfidf.index" ]; then
    echo 'Building FAISS TF-IDF index from tfidf_matrix.npz...'
    python -c "
import faiss, scipy.sparse as sp, numpy as np

print('Loading TF-IDF matrix...')
mat = sp.load_npz('/app/models_store/tfidf_matrix.npz')
dense = mat.toarray().astype('float32')
faiss.normalize_L2(dense)
dim = dense.shape[1]
print(f'Building index (dim={dim})...')
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, 50, faiss.METRIC_INNER_PRODUCT)
index.train(dense)
index.add(dense)
faiss.write_index(index, '/app/models_store/faiss_tfidf.index')
print(f'Done — {index.ntotal} vectors indexed')
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