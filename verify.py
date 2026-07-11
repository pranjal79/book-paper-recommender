import pickle
import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp

# TF-IDF
with open("models_store/tfidf_vectorizer.pkl", "rb") as f:
    vec = pickle.load(f)

print(f"TF-IDF vocabulary size: {len(vec.vocabulary_)}")

# Embeddings
emb = np.load("models_store/sentence_embeddings.npy")
print(f"Embeddings shape: {emb.shape}")

# FAISS
idx = faiss.read_index("models_store/faiss_semantic.index")
print(f"FAISS vectors: {idx.ntotal}")

print("\n✅ Everything looks good!")