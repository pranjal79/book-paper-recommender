from dotenv import load_dotenv
import os

load_dotenv()

print("URI   :", os.getenv("MLFLOW_TRACKING_URI"))
print("User  :", os.getenv("MLFLOW_TRACKING_USERNAME"))

token = os.getenv("MLFLOW_TRACKING_PASSWORD")

if token:
    print("Token : SET ✅")
else:
    print("Token : MISSING ❌")