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