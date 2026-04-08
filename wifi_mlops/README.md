# WiFi Simulation MLOps Pipeline

This project implements a simplified Simulation-based MLOps pipeline running locally.

## Features
- **Simulation**: A 2D simulation environment dropping transmitters and shapes ('circle', 'square', 'triangle') and recording path delays/attenuations (like an FMCW or basic ToF radar).
- **Data Pipeline**: Generates features and labels into memory, normalizes using Min-Max scaling.
- **Model**: A PyTorch MLP (Multi-Layer Perceptron) acting as a lightweight classifier.
- **Serving**: FastAPI service for triggering simulations and predicting shapes from features.

## Project Structure
```text
wifi_mlops/
├── README.md
├── requirements.txt
├── run_pipeline.py    # Master script to trigger training pipeline
├── artifacts/         # Models and prepocessors are saved here
└── src/
    ├── simulation/
    │   └── simulator.py  # Simulation logic
    ├── data/
    │   └── pipeline.py   # Dataset generator and normalization
    ├── model/
    │   ├── network.py    # PyTorch NN definitions
    │   └── train.py      # Training loop and exporting logic
    └── api/
        └── main.py       # FastAPI application
```

## How to Run

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Run Pipeline (Data Gen & Training)**
Run the master script to generate data, train the model, and export artifacts.
```bash
python run_pipeline.py
```
*You will see the model weights (`model.pth`) and preprocessing configs (`preproc.pkl`) saved in `./artifacts`.*

3. **Start the API Server**
Run the FastAPI application via uvicorn:
```bash
uvicorn src.api.main:app --reload
```

4. **Interact with the API**
Go to your browser: http://127.0.0.1:8000/docs
- Use `/simulate` to generate a test sample.
- Use `/predict` by copying the `features` array output from `/simulate` into the prediction payload.
