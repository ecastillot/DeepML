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
from DeepML.model import MultiTaskCNN,DenseNet,CombinedLoss

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

map_path = "/home/edc240000/DeepML/tests/utils/original_map.png"
fig = data.plot_map()
fig.savefig(map_path,dpi=300)

n_events = 2500
n_noise = 2500

noise_mask = create_sample_mask(metadata=data.metadata,category="noise",n_samples=n_noise)
event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
                                n_samples=n_events,min_mag=2)

data.filter(noise_mask | event_mask)


map_path = "/home/edc240000/DeepML/tests/utils/filtered_map.png"
fig = data.plot_map()
fig.savefig(map_path,dpi=300)
# print(data)
exit()
generator = sbg.GenericGenerator(data)
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

@generator.augmentation
def magnitude_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    if metadata["trace_category"] == "noise":
        norm_mag = np.array([[0.0]])
    else:
        norm_mag = normalize_magnitude(metadata["source_magnitude"])
    # print("norm_mag",norm_mag,metadata["trace_category"])
    # print("norm_mag",type(norm_mag),metadata["trace_category"])
    # print("norm_mag",norm_mag.shape,metadata["trace_category"])
    state_dict["y_magnitude"] = norm_mag
    

generator.add_augmentations(
                            [
                            normalize,
                            # p_s_labels,
                            detection_label
                            ]
                        )

train_loader = DataLoader(generator, batch_size=100, 
                          shuffle=True, num_workers=4)
# model = MultiTaskCNN()
model = DenseNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# loss_fn = nn.MSELoss()
loss_fn = CombinedLoss(detection_weight=1.0, magnitude_weight=1.0)

for epoch in range(10):
    model.train()
    epoch_loss = 0.0
    det_loss_total = 0.0
    mag_loss_total = 0.0
    num_batches = 0
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
        
        print(num_batches,num_batches*100,epoch_loss)
        
        det_loss_total += loss_det
        mag_loss_total += loss_mag
        num_batches += 1
    
    print(f"Epoch {epoch + 1} | Total Loss: {epoch_loss / num_batches:.4f} | "
          f"Detection Loss: {det_loss_total / num_batches:.4f} | "
          f"Magnitude Loss: {mag_loss_total / num_batches:.4f}")