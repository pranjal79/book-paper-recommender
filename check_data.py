import pandas as pd

books = pd.read_csv("data/raw/books_raw.csv")
papers = pd.read_csv("data/raw/papers_raw.csv")

print("=== BOOKS ===")
print(f"Shape     : {books.shape}")
print(f"Columns   : {list(books.columns)}")
print(f"Nulls     :\n{books.isnull().sum()}")
print(books.head(2).to_string())

print("\n=== PAPERS ===")
print(f"Shape     : {papers.shape}")
print(f"Columns   : {list(papers.columns)}")
print(f"Nulls     :\n{papers.isnull().sum()}")
print(papers.head(2).to_string())