import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from src.model.train import train_model

if __name__ == '__main__':
    print("="*50)
    print("🚀 Welcome to the WiFi Simulation MLOps Pipeline 🚀")
    print("="*50)
    print("\n[Step 1] Running Data Generation & Model Training...")
    train_model()
    
    print("\n[Step 2] Artifacts have been generated. To start the inference server, run the following command:")
    print("         uvicorn src.api.main:app --reload\n")
    print("Then open your browser at: http://127.0.0.1:8000/docs\n")
