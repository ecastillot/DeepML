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
from DeepML.utils import create_sample_mask,prepare_data_generators,get_scalar_magnitude_predictions


data = sbd.TXED()

n_events = 5000

event_mask = create_sample_mask(metadata=data.metadata,category="earthquake_local",
                                n_samples=n_events,min_mag=-2,random_state=42)

data.filter( event_mask)

magnitude_scaler = "/home/edc240000/DeepML/output/scaler/magnitude_scaler.pkl"
generators = prepare_data_generators(data=data,scaler_path=magnitude_scaler )
train_loader = DataLoader(generators["generator_train"], batch_size=100, shuffle=True)
val_loader = DataLoader(generators["generator_dev"], batch_size=100, shuffle=False)
test_loader = DataLoader(generators["generator_test"], batch_size=100, shuffle=False)


save_dir = "/home/edc240000/DeepML/output/model_outputs/magnitude/"  # folder where you want to save the results



model_classes = {
    "Perceptron": Perceptron,
    "DNN": DNN,
    "CNNSE": CNNSE,
    "CNNDE": CNNDE,
}

model_paths = dict((x, f"/home/edc240000/DeepML/output/models/magnitude/{x}/best/best_model_{x}.pt") for x in model_classes.keys())

predictions = get_scalar_magnitude_predictions(model_classes=model_classes,
                                 model_paths=model_paths,
                                 data_loader=test_loader,
                                 save_dir=save_dir,
                                 load_y=True)