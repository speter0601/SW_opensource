import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.pipeline import generate_dataset
from model.network import WiFiShapePredictor

def train_model():
    # 1. Data Generation & Preprocessing
    X, y, X_min, diff = generate_dataset(2000)
    
    # Save preprocessing parameters in 'artifacts' dir relative to root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    artifacts_dir = os.path.join(project_root, 'artifacts')
    os.makedirs(artifacts_dir, exist_ok=True)
    
    preproc_path = os.path.join(artifacts_dir, 'preproc.pkl')
    joblib.dump({'X_min': X_min, 'diff': diff}, preproc_path)
    
    # 2. Train / Val Split
    indices = torch.randperm(len(X))
    split = int(0.8 * len(X))
    
    train_idx = indices[:split]
    val_idx = indices[split:]
    
    X_train, y_train = torch.FloatTensor(X[train_idx]), torch.LongTensor(y[train_idx])
    X_val, y_val = torch.FloatTensor(X[val_idx]), torch.LongTensor(y[val_idx])
    
    # 3. Model Setup
    model = WiFiShapePredictor(input_dim=X.shape[1], num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Training Loop
    epochs = 50
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            # 5. Evaluation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, predicted = torch.max(val_outputs, 1)
                accuracy = (predicted == y_val).sum().item() / len(y_val)
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {accuracy:.4f}")
            
    # 6. Artifact Saving
    model_path = os.path.join(artifacts_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model and parameters exported to '{artifacts_dir}'.")

if __name__ == '__main__':
    train_model()
