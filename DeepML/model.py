import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 6000, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 6000)

        self.det_out = nn.Sigmoid()
        self.mag_out = nn.Linear(512, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        hidden = F.relu(self.fc2(x))
        det = self.det_out(self.fc3(hidden)).unsqueeze(1)
        mag = self.mag_out(hidden)
        return det, mag

class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),  # -> (batch, 16, 1500)
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(4),  # -> (batch, 32, 375)
        )
        
        self.detection_head = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1),  # -> (batch, 1, 375)
            nn.Upsample(scale_factor=16, mode="linear", align_corners=False),  # back to 6000 if needed
            nn.Sigmoid(),  # -> (batch, 1, 6000)
        )
        
        self.magnitude_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 32, 1)
            nn.Flatten(),             # -> (batch, 32)
            nn.Linear(32, 1),         # -> (batch, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        y_pred = self.detection_head(x)  # shape: (batch, 1, 6000) after upsampling
        m_pred = self.magnitude_head(x)  # shape: (batch, 1)
        return y_pred, m_pred
    
class CombinedLoss(nn.Module):
    def __init__(self, detection_weight=1.0, magnitude_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.dw = detection_weight
        self.mw = magnitude_weight

    def forward(self, y_pred_det, y_true_det, y_pred_mag, y_true_mag):
        loss_det = self.bce(y_pred_det, y_true_det)

        # Apply magnitude loss only if detection is present in ground truth
        mask = (y_true_det.max(dim=-1).values > 0).float().view(-1, 1)  # shape: (batch_size, 1)
        loss_mag = self.mse(y_pred_mag * mask, y_true_mag * mask)

        return self.dw * loss_det + self.mw * loss_mag, loss_det.item(), loss_mag.item()

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__() 
        self.bce = nn.BCELoss()
    
    def forward(self, y_pred_det, y_true_det):
        loss_det = self.bce(y_pred_det, y_true_det)
        return loss_det
    
class CNNDetection(nn.Module):
    def __init__(self,input_length=6000,num_classes=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=11, padding="same"),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 8, input_length/2) e.g., (batch, 8, 3000)
            nn.Conv1d(8, 16, kernel_size=9, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 16, input_length/4) e.g., (batch, 16, 1500)
            nn.Conv1d(16, 16, kernel_size=7, padding="same"),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 16, input_length/8) e.g., (batch, 16, 750)
            nn.Conv1d(16, 32, kernel_size=7, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 32, input_length/16) e.g., (batch, 32, 375)
            nn.Conv1d(32, 32, kernel_size=5, padding="same"),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 32, input_length/32) e.g., (batch, 32, 187)
            nn.Conv1d(32, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 64, input_length/64) e.g., (batch, 64, 93)
            nn.Conv1d(64, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # -> (batch, 64, input_length/128) e.g., (batch, 64, 46)
        )
        
        # Calculate the flattened size after the encoder
        self.flattened_size = 64 * (input_length // 128)  # e.g., 64 * 46 = 2944
        
        self.dense = nn.Sequential(
            nn.Flatten(),  # Flatten the output of the encoder
            nn.Linear(self.flattened_size, 128),  # First dense layer
            nn.ReLU(),
            nn.Dropout(0.5),  # Optional: Dropout for regularization
            nn.Linear(128, 32),  # Second dense layer
            nn.ReLU(),
            nn.Linear(32, num_classes),  # Output layer (e.g., 2 for binary classification)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.encoder(x)  # Pass through convolutional layers
        x = self.dense(x)    # Pass through dense layers
        return x

    # def forward(self, x):
    #     x = self.encoder(x)
    #     y_pred = self.detection_head(x)  # shape: (batch, 1, 6000) after upsampling
    #     m_pred = self.magnitude_head(x)  # shape: (batch, 1)
    #     return y_pred, m_pred