import torch
import torch.nn as nn
import torch.nn.functional as F


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

        return self.dw * loss_det + self.mw * loss_mag
    
class DenseNet(nn.Module):
    def __init__(self):
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
    
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.detector = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        feat = self.conv(x)
        det = self.detector(feat)  # (batch, 1, T)
        mag = self.regressor(feat)  # (batch, 1)
        return det, mag
    
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Conv1d(3, 16, kernel_size=7, padding=3)
        self.res1 = ResidualBlock(16, 32)
        self.res2 = ResidualBlock(32, 64)

        self.detector = nn.Sequential(
            nn.Conv1d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = F.relu(self.initial(x))
        x = self.res1(x)
        x = self.res2(x)
        det = self.detector(x)
        mag = self.regressor(x)
        return det, mag
    
class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.flatten = nn.Flatten()
        self.fc_branch = nn.Sequential(
            nn.Linear(16 * 3000, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        self.detector = nn.Sequential(
            nn.Linear(256, 6000),
            nn.Sigmoid()
        )

        self.regressor = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv_branch(x)
        x = self.flatten(x)
        x = self.fc_branch(x)
        det = self.detector(x).unsqueeze(1)
        mag = self.regressor(x)
        return det, mag