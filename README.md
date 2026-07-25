---
title: Book & Research Paper Recommender
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# 📚 Book & Research Paper Recommendation System

An end-to-end **Machine Learning & NLP-based Recommendation System** that suggests similar **books** and **research papers** based on a user's query using **TF-IDF**, **Sentence Transformers**, and **FAISS** for fast similarity search.

## 🚀 Live Demo

🌐 **Streamlit App:**  
https://book-paper-recommender-vfrgmwacpiptapqmnlree4.streamlit.app/

## 📸 Preview

> Search for a book title, research paper title, or abstract and receive highly relevant recommendations using multiple similarity techniques.

---

# ✨ Features

- 📚 Recommend similar Books
- 📄 Recommend similar Research Papers
- 🔍 TF-IDF (Keyword-based Search)
- 🧠 Semantic Search using Sentence Transformers
- ⚡ Fast Similarity Search with FAISS
- ⚖️ Compare TF-IDF and Semantic Results
- 🎛️ Interactive Streamlit Dashboard
- 📊 Adjustable Number of Recommendations
- 📂 Search Books, Papers, or Both

---

# 🛠️ Tech Stack

## Machine Learning & NLP

- Python
- Scikit-learn
- TF-IDF Vectorizer
- Sentence Transformers
- FAISS
- NLTK
- spaCy

## Data Engineering

- Pandas
- NumPy
- DVC (Data Version Control)

## Experiment Tracking

- MLflow
- DagsHub

## Deployment

- Streamlit Community Cloud
- Docker
- GitHub Actions

---

# 📂 Project Structure

```
book-paper-recommender/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│   └── utils/
│
├── data/
│
├── models_store/
│
├── notebooks/
│
├── requirements.txt
├── Dockerfile
├── dvc.yaml
├── params.yaml
└── README.md
```

---

# 🚀 Recommendation Methods

## 1️⃣ TF-IDF Search

- Keyword-based similarity
- Fast and lightweight
- Best for exact title matching

## 2️⃣ Semantic Search

- Sentence Transformer Embeddings
- Understands contextual meaning
- Better recommendations even with different wording

## 3️⃣ Compare Both

Displays results from both approaches side by side for easy comparison.

---

# 📊 Dataset

The project uses a combined dataset containing:

- 📚 Books
- 📄 Research Papers

After preprocessing, the data is cleaned, normalized, and indexed for efficient retrieval.

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/pranjal79/book-paper-recommender.git

cd book-paper-recommender
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

# 🧠 Model Pipeline

```
Raw Data
      │
      ▼
Data Cleaning
      │
      ▼
Text Preprocessing
      │
      ▼
TF-IDF Vectorization
      │
      ├────────► FAISS Index
      │
Sentence Transformer Embeddings
      │
      ├────────► FAISS Semantic Index
      │
      ▼
Streamlit Recommendation System
```

---

# 📈 MLOps Tools Used

- ✅ MLflow
- ✅ DVC
- ✅ GitHub Actions
- ✅ Docker
- ✅ Streamlit Community Cloud

---

# 💡 Future Improvements

- User Authentication
- Personalized Recommendations
- Search History
- Book Cover Images
- PDF Preview for Research Papers
- Advanced Filtering
- Hybrid Ranking Algorithm
- REST API using FastAPI
- Cloud Database Integration

---

# 👨‍💻 Author

**Pranjal Panigrahi**

GitHub:  
https://github.com/pranjal79

LinkedIn: *(Add your LinkedIn profile link here)*

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub!

---

## 📄 License

This project is licensed under the MIT License.