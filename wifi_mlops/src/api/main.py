from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional
import torch
import joblib
import os
import sys

# Ensure imports work regardless of execution context
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.simulator import WiFiSimulator, MATERIALS
from model.network import WiFiShapePredictor

app = FastAPI(
    title="WiFi Sim MLOps API",
    description="API for simulating WiFi reflection and predicting materials.",
    version="1.0"
)

simulator = WiFiSimulator()
model = None
preproc = None

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
artifacts_dir = os.path.join(project_root, 'artifacts')

class PredictionRequest(BaseModel):
    features: List[float]

def load_artifacts():
    global model, preproc
    model_path = os.path.join(artifacts_dir, "model.pth")
    preproc_path = os.path.join(artifacts_dir, "preproc.pkl")
    if os.path.exists(model_path) and os.path.exists(preproc_path):
        preproc = joblib.load(preproc_path)
        model = WiFiShapePredictor()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return True
    return False

@app.on_event("startup")
def on_startup():
    if load_artifacts():
        print("ML Artifacts loaded.")
    else:
        print("WARNING: ML Artifacts not found. Please run the training pipeline.")

html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>WiFi MLOps</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; text-align: center; }
        button { padding: 10px 20px; font-size: 16px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }
        #result { margin-top: 20px; font-size: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h1>WiFi ML Simulation Dashboard</h1>
    <button onclick="runSimulation()">Run Simulation & Predict</button>
    <div id="result">Waiting for simulation...</div>
    <canvas id="signalChart" width="800" height="400" style="margin-top: 30px;"></canvas>

    <script>
        let chart;
        async function runSimulation() {
            document.getElementById("result").innerText = "Simulating...";
            
            // 1. Simulate
            const simRes = await fetch('/simulate', {method: 'POST'});
            const simData = await simRes.json();
            const features = simData.data.features;
            const trueMaterial = simData.data.label_name;
            
            // 2. Predict
            const predRes = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({features: features})
            });
            const predData = await predRes.json();
            
            document.getElementById("result").innerText = 
                "True: " + trueMaterial + " | Predicted: " + predData.predicted_material;

            // 3. Plot signal
            const ctx = document.getElementById('signalChart').getContext('2d');
            if (chart) chart.destroy();
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: features.length}, (_, i) => i),
                    datasets: [{
                        label: 'Received Signal Waveform',
                        data: features,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                }
            });
        }
    </script>
</body>
</html>
"""

@app.get("/")
def get_ui():
    return HTMLResponse(content=html_content)

@app.post("/simulate")
def simulate(material: Optional[str] = None):
    """
    Generate a simulation round. Optionally specify a material.
    """
    if material and material not in MATERIALS:
        raise HTTPException(status_code=400, detail=f"Invalid material. Choose from {MATERIALS}")
    res = simulator.simulate(material)
    return {"status": "success", "data": res}

@app.post("/predict")
def predict(req: PredictionRequest):
    """
    Predict the material given a feature array (100 elements).
    """
    if model is None or preproc is None:
        if not load_artifacts():
            raise HTTPException(status_code=500, detail="Model artifacts not found. Run training first.")
    
    import numpy as np
    features = np.array(req.features)
    
    if features.shape[0] != 100:
        raise HTTPException(status_code=400, detail=f"Expected feature size 100, got {features.shape[0]}")
    
    # Preprocessing
    features_norm = (features - preproc['X_min']) / preproc['diff']
    
    # Inference
    with torch.no_grad():
        x_tensor = torch.FloatTensor(features_norm).unsqueeze(0)
        outputs = model(x_tensor)
        _, predicted = torch.max(outputs, 1)
        
    material_idx = predicted.item()
    return {
        "predicted_material": MATERIALS[material_idx],
        "class_index": material_idx,
    }

@app.get("/plot")
def plot_signal():
    result = simulator.simulate()
    features = result["features"]

    plt.figure()
    plt.plot(features)
    plt.title("WiFi Signal (Simulated Features)")
    plt.xlabel("Bins")
    plt.ylabel("Amplitude")

    filepath = "/tmp/plot.png"
    plt.savefig(filepath)
    plt.close()

    return FileResponse(filepath)
