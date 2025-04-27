import sys
path = "/home/edc240000/DeepML"
sys.path.append(path)

# ##### tx dataset #####
import os
import math
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


from DeepML.scalar_magnitude.models import CNNSE,CNNDE,DNN,Perceptron,MagnitudeLoss
from DeepML.utils import create_sample_mask, load_or_init_history, handle_checkpoint,prepare_data_generators


data = sbd.TXED()

n_events = 5000
event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
                                n_samples=n_events,min_mag=-2,random_state=42)

data.filter(event_mask)

magnitude_scaler = "/home/edc240000/DeepML/output/scaler/magnitude_scaler.pkl"

generators = prepare_data_generators(data=data,scaler_path=magnitude_scaler )
# print(generators["generator_train"][0])
train_loader = DataLoader(generators["generator_train"], batch_size=100, shuffle=True)
val_loader = DataLoader(generators["generator_dev"], batch_size=100, shuffle=False)
test_loader = DataLoader(generators["generator_test"], batch_size=100, shuffle=False)


# print(len(generators["generator_train"])*100/5000)
# print(len(generators["generator_dev"])*100/5000)
# print(len(generators["generator_test"])*100/5000)
# exit()

model_1 = CNNSE()
model_2 = CNNDE()
model_3 = DNN()
model_4 = Perceptron()

models = [model_1,model_2,model_3,model_4]

for model in models:

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = nn.MSELoss()
    loss_fn = MagnitudeLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=5,
                                                        verbose=True)
    # dev_batches = 0
    best_dev_loss = float('inf')
    patience = 10
    epochs_no_improve = 0

    arch_path = f"/home/edc240000/DeepML/output/models/magnitude/{model.name}/architecture/"
    os.makedirs(os.path.dirname(arch_path),exist_ok=True)
    model.export_architecture(arch_path)

    checkpoint_path = f"/home/edc240000/DeepML/output/models/magnitude/{model.name}/best/best_model_{model.name}.pt"
    os.makedirs(os.path.dirname(checkpoint_path),exist_ok=True)
    checkpoint_path = handle_checkpoint(checkpoint_path)

    history_file = f"/home/edc240000/DeepML/output/models/magnitude/{model.name}/best/training_history_{model.name}.json"
    os.makedirs(os.path.dirname(history_file),exist_ok=True)

    history = load_or_init_history(history_file)

    for epoch in range(50):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        train_batches = 0
        
        # Initialize metrics for the epoch
        total_mae = 0.0
        total_mse = 0.0
        total_rmse = 0.0
        
        
        for batch in train_loader:
            
        #     exit()
            x = batch["X"].to(dtype=torch.float32)
            y = batch["y_scalar_magnitude"].to(dtype=torch.float32)
            
            # print("X",x.shape)
            # print("y",y.shape)
            
            y_pred = model(x)
            # print("y",y.shape)
            # print("y_pred",y_pred.shape)
            # exit()
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            epoch_loss += loss.item()
            
            # Calculate MAE, MSE, RMSE
            mae = F.l1_loss(y_pred, y, reduction='sum').item()
            mse = F.mse_loss(y_pred, y, reduction='sum').item()
            rmse = math.sqrt(mse)
            
            total_mse += mse
            total_mae += mae
            total_rmse += rmse
            
            # print(train_batches,train_batches*100,epoch_loss)
            
            train_batches += 1
        
        avg_train_loss = epoch_loss / train_batches
        avg_train_mse = total_mse / (train_batches * y.numel())
        avg_train_mae = total_mae / (train_batches * y.numel())
        avg_train_rmse = total_rmse / (train_batches * y.numel())
        
        
        model.eval()
        dev_loss = 0.0
        dev_batches = 0
        
        total_mae = 0.0
        total_mse = 0.0
        total_rmse = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["X"].to(dtype=torch.float32)
                y = batch["y_scalar_magnitude"].to(dtype=torch.float32)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)

                dev_loss += loss.item()
                
                # Calculate MAE, MSE, RMSE
                mae = F.l1_loss(y_pred, y, reduction='sum').item()
                mse = F.mse_loss(y_pred, y, reduction='sum').item()
                rmse = math.sqrt(mse)
                
                total_mse += mse
                total_mae += mae
                total_rmse += rmse


                dev_batches += 1
        
        avg_dev_loss = dev_loss / dev_batches
        avg_dev_mse = total_mse / (dev_batches * y.numel())
        avg_dev_mae = total_mae / (dev_batches * y.numel())
        avg_dev_rmse = total_rmse / (dev_batches * y.numel())

        end_time = time.time()
        epoch_time = end_time - start_time
        if not history["acum_time"]:
            acum_time = epoch_time
        else:
            acum_time = history["acum_time"][-1]+epoch_time
        
        history["train_loss"].append(avg_train_loss)
        history["dev_loss"].append(avg_dev_loss)
        history["train_mse"].append(avg_train_mse)
        history["train_mae"].append(avg_train_mae)
        history["train_rmse"].append(avg_train_rmse)
        history["dev_mse"].append(avg_dev_mse)
        history["dev_mae"].append(avg_dev_mae)
        history["dev_rmse"].append(avg_dev_rmse)
        history["acum_time"].append(acum_time)

        # print(f"Epoch [{epoch+1}/50] - Train Loss: {avg_train_loss:.4f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}")

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
            f"Dev Loss: {avg_dev_loss:.4f} | "
            f"Train MSE: {avg_train_mse:.4f} | Dev MSE: {avg_dev_mse:.4f} | "
            f"Train MAE: {avg_train_mae:.4f} | Dev MAE: {avg_dev_mae:.4f} | "
            f"Train RMSE: {avg_train_rmse:.4f} | Dev RMSE: {avg_dev_rmse:.4f} | "
            f"Acum Time: {acum_time:.2f}")
        
        with open(history_file, "w") as f:
            json.dump(history, f)
        
        # Scheduler step
        scheduler.step(avg_dev_loss)

        torch.save(model.state_dict(), checkpoint_path)
        # # Early stopping + model saving
        # if avg_dev_loss < best_dev_loss:
        #     print("\u2705 New best model. Saving...") #  \u2705  good check emoticon
        #     best_dev_loss = avg_dev_loss
        #     torch.save(model.state_dict(), checkpoint_path)
        #     epochs_no_improve = 0
        # else:
        #     epochs_no_improve += 1
        #     print(f"\u26A0  No improvement for {epochs_no_improve} epochs.")
        #     # print(f"\u274C  No improvement for {epochs_no_improve} epochs.")

        #     if epochs_no_improve >= patience:
        #         print("\U0001F6D1 Early stopping triggered.") # \U0001F6D1 stopped emoticon
        #         break
        
