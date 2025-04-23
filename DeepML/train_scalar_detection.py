import sys
path = "/home/edc240000/DeepML"
sys.path.append(path)

# ##### tx dataset #####
import os
root = "/groups/igonin/.seisbench"
os.environ["SEISBENCH_CACHE_ROOT"] = root
import joblib
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
from utils import create_sample_mask, load_or_init_history, handle_checkpoint


data = sbd.TXED()

# map_path = "/home/edc240000/DeepML/tests/utils/original_map.png"
# fig = data.plot_map()
# fig.savefig(map_path,dpi=300)

n_events = 2500
n_noise = 2500

noise_mask = create_sample_mask(metadata=data.metadata,category="noise",n_samples=n_noise)
event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
                                n_samples=n_events,min_mag=2)

data.filter(noise_mask | event_mask)

train, dev, test = data.train_dev_test()
# print(train,dev,test)
# exit()

# map_path = "/home/edc240000/DeepML/tests/utils/filtered_map.png"
# fig = data.plot_map()
# fig.savefig(map_path,dpi=300)
# print(data)
# exit()
generator_train = sbg.GenericGenerator(train)
generator_dev = sbg.GenericGenerator(dev)
generator_test = sbg.GenericGenerator(test)
# print(generator)

normalize = sbg.Normalize(detrend_axis=0,
                            amp_norm_type="peak",
                            eps=1e-8,
                            key = "X")
detection_label = sbg.DetectionLabeller(
    p_phases="trace_p_arrival_sample",
    s_phases="trace_s_arrival_sample",
    factor=1.5,
    key=("X","y_detection"))
p_s_labels = sbg.ProbabilisticLabeller(
                    label_columns=["trace_p_arrival_sample",
                                   "trace_s_arrival_sample"],
                    sigma=50,
                    dim=-2,
                    key=("X","y_picks")
                    )


scaler_path = "/home/edc240000/DeepML/tests/magnitude_scaler.pkl"
scaler = joblib.load(scaler_path)


def normalize_magnitude(magnitude: float):
    # magnitude = np.array([magnitude]).reshape(-1, 1)
    magnitude_df = pd.DataFrame([[magnitude]], columns=["source_magnitude"])
    return scaler.transform(magnitude_df)

def denormalize_magnitude(magnitude: np.ndarray):
    # magnitude = np.array(magnitude).reshape(-1, 1)
    return scaler.inverse_transform(magnitude).flatten()

@generator_train.augmentation
def magnitude_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    if metadata["trace_category"] == "noise":
        norm_mag = np.array([[0.0]])
    else:
        norm_mag = normalize_magnitude(metadata["source_magnitude"])

    state_dict["y_magnitude"] = norm_mag

@generator_dev.augmentation
def magnitude_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    if metadata["trace_category"] == "noise":
        norm_mag = np.array([[0.0]])
    else:
        norm_mag = normalize_magnitude(metadata["source_magnitude"])

    state_dict["y_magnitude"] = norm_mag
    
@generator_test.augmentation
def magnitude_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    if metadata["trace_category"] == "noise":
        norm_mag = np.array([[0.0]])
    else:
        norm_mag = normalize_magnitude(metadata["source_magnitude"])

    state_dict["y_magnitude"] = norm_mag

@generator_train.augmentation
def event_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    
    if metadata["trace_category"] == "noise":
        y = np.array([0])
    else:
        y = np.array([1])
    
    state_dict["y_scalar_detection"] = [y.reshape(1, 1)]
    
@generator_test.augmentation
def event_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    
    if metadata["trace_category"] == "noise":
        y = np.array([0])
    else:
        y = np.array([1])
    
    state_dict["y_scalar_detection"] = [y.reshape(1, 1)]
    

@generator_dev.augmentation
def event_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    
    if metadata["trace_category"] == "noise":
        y = np.array([0])
    else:
        y = np.array([1])
    
    state_dict["y_scalar_detection"] = [y.reshape(1, 1)]

for g in [generator_train, generator_dev, generator_test]:
    g.add_augmentations(
                        [
                        normalize,
                        # p_s_labels,
                        detection_label
                        ]
                    )
# print(generator_train[0])
# print(generator_test[0])
# print(generator_dev[0])
# exit()

train_loader = DataLoader(generator_train, batch_size=100, shuffle=True)
val_loader = DataLoader(generator_test, batch_size=100, shuffle=False)
test_loader = DataLoader(generator_test, batch_size=100, shuffle=False)


# print(train_loader)
# print(val_loader)
# print(test_loader)
# exit()

model = CNNDetection()
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

checkpoint_path = "/home/edc240000/DeepML/DeepML/best_model.pt"
checkpoint_path = handle_checkpoint(checkpoint_path)

history_file = "/home/edc240000/DeepML/DeepML/training_history.json"
history = load_or_init_history(history_file)

for epoch in range(100):
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

    
    print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
          f"Dev Loss: {avg_dev_loss:.4f} | "
          f"Train Acc: {train_accuracy:.4f} | Dev Acc: {dev_accuracy:.4f}")
    
    history["train_loss"].append(avg_train_loss)
    history["dev_loss"].append(avg_dev_loss)
    history["train_acc"].append(train_accuracy)
    history["dev_acc"].append(dev_accuracy)

    
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
        
