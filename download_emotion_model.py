import os
import os
import gdown

MODEL_URL = "https://drive.google.com/uc?id=1EyQPiojTvsrNpDIPKYGC6gM71kTZ5mMp"
MODEL_PATH = os.path.join("emotion_bert_model", "model.safetensors")

os.makedirs("emotion_bert_model", exist_ok=True)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model from {MODEL_URL}...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download complete.")
    else:
        print("Model file already exists.")

if __name__ == "__main__":
    download_model()
    print("Model file already exists.")
