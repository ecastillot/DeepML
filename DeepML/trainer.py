from torch.utils.data import DataLoader

import os
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from DeepML.model import MultiTaskCNN

data = sbd.TXED()
generator = sbg.GenericGenerator(data)
# print(generator)
# exit()

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
    magnitude = np.array([magnitude]).reshape(-1, 1)
    print(magnitude)
    return scaler.transform(magnitude)

def denormalize_magnitude(magnitude: np.ndarray):
    # magnitude = np.array(magnitude).reshape(-1, 1)
    return scaler.inverse_transform(magnitude).flatten()

@generator.augmentation
def magnitude_labeler(state_dict):
    waveforms, metadata = state_dict["X"]
    
    norm_mag = normalize_magnitude(metadata["source_magnitude"])
    state_dict["y_magnitude"] = [norm_mag]
    

generator.add_augmentations(
                            [
                            normalize,
                            # p_s_labels,
                            detection_label
                            ]
                        )




# Access a sample (check it’s working)
sample = generator[340055]
print(sample.keys())
train_loader = DataLoader(generator, batch_size=32, 
                          shuffle=True, num_workers=4)

print(train_loader)

model = MultiTaskCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(10):
    model.train()
    for batch in train_loader:
    #     print(batch["X"])
    #     exit()
        x = batch["X"]
        m = batch["y_magnitude"]
        y = batch["y_detection"]
        
        x = torch.tensor(x, dtype=torch.float32)
        m = torch.tensor(m, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        print("X",x.shape)
        print("m",m.shape)
        print("y",y.shape)
        # mags = batch["y_magnitude"][0]  # shape: (batch_size, 1)
        # print(mags)
#         waveforms = torch.tensor(waveforms, dtype=torch.float32).cuda()  # shape: (B, 3, T)
#         mags = torch.tensor(mags, dtype=torch.float32).cuda()  # shape: (B, 1)

        y_pred,m_pred = model(x)
        print("m_pred",m_pred.shape)
        print("y_pred",y_pred.shape)
        exit()
#         loss = loss_fn(output, mags)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch} | Loss: {loss.item():.4f}")