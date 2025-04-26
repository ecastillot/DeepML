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
from DeepML.model import *
import json
import matplotlib.pyplot as plt
from utils import create_sample_mask, load_or_init_history, handle_checkpoint,prepare_data_generators


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


# print(train_loader)
# print(val_loader)
# print(test_loader)
# exit()

# model = CNNSE()
# model = CNNDE()
# model = DNN()
model = Perceptron()

# model = CNNDetection()
# model = MultiTaskCNN()
# model = DenseNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()
loss_fn = DetectionLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                       factor=0.5, patience=5,
                                                       verbose=True)
# dev_batches = 0
best_dev_loss = float('inf')
patience = 10
epochs_no_improve = 0

arch_path = f"/home/edc240000/DeepML/output/models/detection/{model.name}/architecture/"
os.makedirs(os.path.dirname(arch_path),exist_ok=True)
model.export_architecture(arch_path)

checkpoint_path = f"/home/edc240000/DeepML/output/models/detection/{model.name}/best/best_model_{model.name}.pt"
os.makedirs(os.path.dirname(checkpoint_path),exist_ok=True)
checkpoint_path = handle_checkpoint(checkpoint_path)

history_file = f"/home/edc240000/DeepML/output/models/detection/{model.name}/best/training_history_{model.name}.json"
os.makedirs(os.path.dirname(history_file),exist_ok=True)

history = load_or_init_history(history_file)

for epoch in range(100):
    start_time = time.time()
    model.train()
    epoch_loss = 0.0
    train_batches = 0
    
    dev_loss = 0.0
    val_batches = 0
    
    correct = 0
    total = 0
    
    for batch in train_loader:
        
    #     exit()
        x = batch["X"].to(dtype=torch.float32)
        y = batch["y_scalar_detection"].to(dtype=torch.float32)
        
        # print("X",x.shape)
        # print("y",y.shape)
        
        y_pred = model(x).unsqueeze(-1)
        # print("y_pred",y_pred.shape)
        # exit()
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predicted = (y_pred > 0.5).float()
        correct += (predicted == y).sum().item()
        total += y.numel()
        
        epoch_loss += loss.item()
        
        # print(train_batches,train_batches*100,epoch_loss)
        
        train_batches += 1
    
    avg_train_loss = epoch_loss / train_batches
    train_accuracy = correct / total
    
    model.eval()
    dev_loss = 0.0
    dev_batches = 0
    
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["X"].to(dtype=torch.float32)
            y = batch["y_scalar_detection"].to(dtype=torch.float32)

            y_pred = model(x).unsqueeze(-1)
            loss = loss_fn(y_pred, y)

            dev_loss += loss.item()

            predicted = (y_pred > 0.5).float()
            correct += (predicted == y).sum().item()
            total += y.numel()

            dev_batches += 1
    
    avg_dev_loss = dev_loss / dev_batches
    dev_accuracy = correct / total

    end_time = time.time()
    epoch_time = end_time - start_time
    acum_time = sum(history["acum_time"])+epoch_time
    
    history["train_loss"].append(avg_train_loss)
    history["dev_loss"].append(avg_dev_loss)
    history["train_acc"].append(train_accuracy)
    history["dev_acc"].append(dev_accuracy)
    history["acum_time"].append(acum_time)

    print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
          f"Dev Loss: {avg_dev_loss:.4f} | "
          f"Train Acc: {train_accuracy:.4f} | Dev Acc: {dev_accuracy:.4f} | "
          f"Acum Time: {acum_time:.2f}")
    
    with open(history_file, "w") as f:
        json.dump(history, f)
    
    # Scheduler step
    scheduler.step(avg_dev_loss)

    # Early stopping + model saving
    if avg_dev_loss < best_dev_loss:
        print("\u2705 New best model. Saving...") #  \u2705  good check emoticon
        best_dev_loss = avg_dev_loss
        torch.save(model.state_dict(), checkpoint_path)
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"\u26A0  No improvement for {epochs_no_improve} epochs.")
        # print(f"\u274C  No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= patience:
            print("\U0001F6D1 Early stopping triggered.") # \U0001F6D1 stopped emoticon
            break
        
