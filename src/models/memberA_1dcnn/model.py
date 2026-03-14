import torch
import torch.nn as nn
from src.models.base import BaseSleepModel


class MemberA1DCNN(BaseSleepModel):
    def __init__(self, n_classes=5, input_channels=1, hidden_dims=(64, 128, 256), dropout=0.5):
        super().__init__(n_classes=n_classes)
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.conv_blocks = nn.Sequential(
            nn.Conv1d(self.input_channels, hidden_dims[0], kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dims[2], n_classes)
        )

    def forward(self, x):
        # x shape: [batch, channels, time]
        x = self.conv_blocks(x)
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def get_config(self):
        return {
            'n_classes': self.n_classes,
            'input_channels': self.input_channels,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout
        }
