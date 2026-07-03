import pandas as pd

raw = pd.read_csv("data/processed/combined_raw.csv")
proc = pd.read_csv("data/processed/combined_processed.csv")

print("=== COMBINED RAW ===")
print(f"Shape   : {raw.shape}")
print(f"Sources : {raw['source'].value_counts().to_dict()}")

print("\n=== COMBINED PROCESSED ===")
print(f"Shape   : {proc.shape}")
print(f"Columns : {list(proc.columns)}")
print(f"Sources : {proc['source'].value_counts().to_dict()}")

print("\nSample ORIGINAL text:")
print(proc['text'].iloc[0][:200])

print("\nSample CLEANED text:")
print(proc['text_clean'].iloc[0][:200])