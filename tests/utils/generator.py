import sys
path = "/home/edc240000/DeepML"
sys.path.append(path)

# ##### tx dataset #####
import os
root = "/groups/igonin/.seisbench"
os.environ["SEISBENCH_CACHE_ROOT"] = root
import joblib

import numpy as np
import seisbench.data as sbd
import seisbench.generate as sbg
import matplotlib.pyplot as plt

data = sbd.TXED()
generator = sbg.GenericGenerator(data)
# print(generator)


detection_label = sbg.DetectionLabeller(
    p_phases="trace_p_arrival_sample",
    s_phases="trace_s_arrival_sample",factor=1.5,
    key=("X","y_detection"))
p_s_labels = sbg.ProbabilisticLabeller(
                    label_columns=["trace_p_arrival_sample","trace_s_arrival_sample"],
                    sigma=50,
                    dim=-2,
                    key=("X","y_picks")
                    )

scaler_path = "/home/edc240000/DeepML/tests/magnitude_scaler.pkl"
scaler = joblib.load(scaler_path)

def normalize_magnitude(magnitude: float):
    magnitude = np.array([magnitude]).reshape(-1, 1)
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
                                # p_s_labels,
                            detection_label
                            ]
                        )

sample = generator[340055]

print(generator)
print("Example:", sample)
print("Sample keys:", sample.keys())

fig = plt.figure(figsize=(10, 7))
axs = fig.subplots(2, 1)
axs[0].plot(sample["X"].T)
axs[0].text(
    0.95, 0.95,                # x, y in axis coordinates (0 = left/bottom, 1 = right/top)
    f"magnitude: {denormalize_magnitude(sample['y_magnitude'])[0]}",         # text string
    transform=axs[0].transAxes,   # use axis coordinates
    ha='right',               # horizontal alignment
    va='top',                 # vertical alignment
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # optional: background box
)


axs[1].plot(sample["y_detection"].T)
# axs[1].plot(sample["y_picks"].T)

axs[1].text(
    0.95, 0.95,                # x, y in axis coordinates (0 = left/bottom, 1 = right/top)
    f"magnitude: {sample['y_magnitude'].item()}",         # text string
    transform=axs[1].transAxes,   # use axis coordinates
    ha='right',               # horizontal alignment
    va='top',                 # vertical alignment
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')  # optional: background box
)

# axs[0].text(0.5, 0.5, f"magnitude: {denormalize_magnitude(sample['y_magnitude'])}", 
#             fontsize=12)


path = "/home/edc240000/DeepML/tests/utils/test.png"


plt.savefig(path)