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
from DeepML.DeepML.detection_model import MultiTaskCNN,DenseNet,CombinedLoss
import json
import matplotlib.pyplot as plt

def load_or_init_history(history_path="history.json"):
    """
    Load existing training history or initialize a new one based on user input.

    Parameters:
        history_path (str): Path to the history JSON file.

    Returns:
        dict: A dictionary containing the training and validation history.
    """
    if os.path.exists(history_path):
        user_input = input(f"\n📁 Found existing history at '{history_path}'.\n"
                           "Do you want to [r]esume or [o]verwrite it? (r/o): ").strip().lower()

        if user_input == "r":
            with open(history_path, "r") as f:
                history = json.load(f)
            print("✅ Resuming training with existing history...\n")
        elif user_input == "o":
            history = {
                "train_loss": [],
                "train_det_loss": [],
                "train_mag_loss": [],
                "dev_loss": [],
                "dev_det_loss": [],
                "dev_mag_loss": [],
            }
            print("🗑️  Previous history cleared. Starting fresh...\n")
        else:
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        history = {
            "train_loss": [],
            "train_det_loss": [],
            "train_mag_loss": [],
            "dev_loss": [],
            "dev_det_loss": [],
            "dev_mag_loss": [],
        }
        print("📄 No previous history found. Starting fresh...\n")

    return history

def handle_checkpoint(checkpoint_path):
    """
    Check if a checkpoint file exists and ask the user whether to remove or keep it.

    Parameters:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        str: The checkpoint path (unchanged), or exits if user cancels.
    """
    if os.path.exists(checkpoint_path):
        user_input = input(f"\n📁 Found existing checkpoint at '{checkpoint_path}'.\n"
                           "Do you want to [d]elete it and start fresh, or [k]eep it? (d/k): ").strip().lower()
        
        if user_input == "d":
            os.remove(checkpoint_path)
            print("🗑️  Old checkpoint deleted. Starting fresh...\n")
        elif user_input == "k":
            print("✅ Keeping existing checkpoint...\n")
            exit()
        else:
            print("❌ Invalid choice. Exiting.")
            exit(1)
    else:
        print("📄 No checkpoint found. Training will start from scratch.\n")
    
    return checkpoint_path

def create_sample_mask(metadata: pd.DataFrame, category: str,
                       n_samples: int, random_state: int = 42,
                       min_mag: float = None, max_mag: float = None) -> pd.Series:
    """
    Create a boolean mask for sampling rows from a metadata DataFrame 
    that match a given trace_category, with optional magnitude filtering
    if the category includes the word "earthquake".

    Parameters
    ----------
    metadata : pd.DataFrame
        The metadata DataFrame that includes 'trace_category' and 'mag' columns.
    category : str
        The target trace_category value to filter by (e.g., "earthquake_local").
    n_samples : int
        The number of samples to draw from the filtered subset.
    random_state : int, optional
        The seed for random sampling to ensure reproducibility (default is 42).
    min_mag : float, optional
        Minimum magnitude to include (only applied if "earthquake" in category).
    max_mag : float, optional
        Maximum magnitude to include (only applied if "earthquake" in category).

    Returns
    -------
    pd.Series
        A boolean Series mask aligned with the original metadata index,
        with True values for sampled rows matching the filtering conditions.
    """
    # Step 1: Filter by trace_category
    mask = metadata["trace_category"] == category
    filtered_metadata = metadata[mask]

    # Step 2: Apply magnitude filtering if "earthquake" is in the category
    if "earthquake" in category.lower():
        if min_mag is not None:
            filtered_metadata = filtered_metadata[filtered_metadata["source_magnitude"] >= min_mag]
        if max_mag is not None:
            filtered_metadata = filtered_metadata[filtered_metadata["source_magnitude"] <= max_mag]

    # Step 3: Sample rows after filtering
    sampled_metadata = filtered_metadata.sample(n=n_samples, random_state=random_state)

    # Step 4: Build final mask from sampled indices
    sampled_indices = sampled_metadata.index
    final_mask = metadata.index.isin(sampled_indices)

    return final_mask


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

model = MultiTaskCNN()
# model = DenseNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()
loss_fn = CombinedLoss(detection_weight=1.0, magnitude_weight=1.0)
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
    train_det_loss = 0.0
    train_mag_loss = 0.0
    train_batches = 0
    
    dev_loss, dev_det_loss, dev_mag_loss = 0.0, 0.0, 0.0
    val_batches = 0
    
    for batch in train_loader:
        
    #     exit()
        x = batch["X"].to(dtype=torch.float32)
        m = batch["y_magnitude"].to(dtype=torch.float32)
        y = batch["y_detection"].to(dtype=torch.float32)
        
        # print("X",x.shape)
        # print("m",m.shape)
        # print("y",y.shape)

        y_pred,m_pred = model(x)
        # print("m_pred",m_pred.shape)
        # print("y_pred",y_pred.shape)
        loss, loss_det, loss_mag = loss_fn(y_pred, y, m_pred, m)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # print(train_batches,train_batches*100,epoch_loss)
        
        train_det_loss += loss_det
        train_mag_loss += loss_mag
        train_batches += 1
    
    model.eval()
    dev_loss = 0.0
    dev_det_loss = 0.0
    dev_mag_loss = 0.0
    dev_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["X"].to(dtype=torch.float32)
            m = batch["y_magnitude"].to(dtype=torch.float32)
            y = batch["y_detection"].to(dtype=torch.float32)

            y_pred, m_pred = model(x)
            loss, loss_det, loss_mag = loss_fn(y_pred, y, m_pred, m)

            dev_loss += loss.item()
            dev_det_loss += loss_det
            dev_mag_loss += loss_mag
            dev_batches += 1
    
    avg_train_loss = epoch_loss / train_batches
    avg_train_det_loss = train_det_loss / train_batches
    avg_train_mag_loss = train_mag_loss / train_batches
    avg_dev_loss = dev_loss / dev_batches
    avg_dev_det_loss = dev_det_loss / dev_batches
    avg_dev_mag_loss = dev_mag_loss / dev_batches
    
    print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} "
          f"(Det: {avg_train_det_loss:.4f}, Mag: {avg_train_mag_loss:.4f}) | "
          f"Dev Loss: {avg_dev_loss:.4f} "
          f"(Det: { avg_dev_det_loss:.4f}, Mag: {avg_dev_mag_loss:.4f})")
    
    history["train_loss"].append(avg_train_loss)
    history["train_det_loss"].append(avg_train_det_loss)
    history["train_mag_loss"].append(avg_train_mag_loss)

    # dev loop...
    history["dev_loss"].append(avg_dev_loss)
    history["dev_det_loss"].append(avg_dev_det_loss)
    history["dev_mag_loss"].append(avg_dev_mag_loss)
    
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
        
