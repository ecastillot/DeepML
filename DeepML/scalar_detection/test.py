import sys
path = "/home/edc240000/DeepML"
sys.path.append(path)

# ##### tx dataset #####
import os
root = "/groups/igonin/.seisbench"
os.environ["SEISBENCH_CACHE_ROOT"] = root
import joblib
import time
import pandas as pd
import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

from DeepML.scalar_detection.models import CNNSE,CNNDE,DNN,Perceptron,DetectionLoss
from DeepML.utils import create_sample_mask,prepare_data_generators


data = sbd.TXED()

n_events = 2500
n_noise = 2500

noise_mask = create_sample_mask(metadata=data.metadata,category="noise",
                                n_samples=n_noise,random_state=42)
event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
                                n_samples=n_events,min_mag=2,random_state=42)

data.filter(noise_mask | event_mask)

magnitude_scaler = "/home/edc240000/DeepML/output/scaler/magnitude_scaler.pkl"
generators = prepare_data_generators(data=data,scaler_path=magnitude_scaler )
train_loader = DataLoader(generators["generator_train"], batch_size=100, shuffle=True)
val_loader = DataLoader(generators["generator_dev"], batch_size=100, shuffle=False)
test_loader = DataLoader(generators["generator_test"], batch_size=100, shuffle=False)

label_p = "Perceptron"
model_path_p = f"/home/edc240000/DeepML/output/models/detection/{label_p}/best/best_model_{label_p}.pt"
label_dnn = "DNN"
model_path_dnn = f"/home/edc240000/DeepML/output/models/detection/{label_dnn}/best/best_model_{label_dnn}.pt"
label_cnnse = "CNNSE"
model_path_cnnse = f"/home/edc240000/DeepML/output/models/detection/{label_cnnse}/best/best_model_{label_cnnse}.pt"
label_cnnde = "CNNDE"
model_path_cnnde = f"/home/edc240000/DeepML/output/models/detection/{label_cnnde}/best/best_model_{label_cnnde}.pt"

save_dir = "/home/edc240000/DeepML/output/model_outputs/detection/"  # folder where you want to save the results



model_classes = {
    "Perceptron": Perceptron,
    "DNN": DNN,
    "CNNSE": CNNSE,
    "CNNDE": CNNDE,
}

model_paths = {
    "Perceptron": model_path_p,
    "DNN": model_path_dnn,
    "CNNSE": model_path_cnnse,
    "CNNDE": model_path_cnnde,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



for label, model_class in model_classes.items():
    print(f"Loading model: {label}")
    
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_paths[label], map_location=device))
    model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in train_loader:
            x = batch["X"].to(dtype=torch.float32)
            y = batch["y_scalar_detection"].to(dtype=torch.float32)
            
            y_pred = model(x).unsqueeze(-1)
            all_preds.append(y_pred)
            all_targets.append(y)

    # Concatenate results
    predictions = torch.cat(all_preds)
    true_labels = torch.cat(all_targets)
    print(f"y_pred shape: {predictions.shape} | y shape: {true_labels.shape}.")
    
    save_path = os.path.join(save_dir,label, f"{label}_outputs.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "y_pred": predictions,
        "y": true_labels
    }, save_path)

    print(f"{label} - Done with predictions.")
    print(f"{label} - Predictions saved to {save_path}\n")
