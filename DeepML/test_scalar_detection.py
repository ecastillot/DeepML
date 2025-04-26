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

