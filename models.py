"""
Model architectures (paper Section 3, 'Implementation'):
'a shared model backbone composed of CNN, TCN, and LSTM layers'.

FeatureExtractor: R^{T x D} -> R^{z}   (f_theta)
Classifier:       R^{z} -> R^{K}       (h_phi)
"""

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    """A single dilated causal-conv TCN block with residual connection."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                                padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.chomp = padding  # amount to trim to keep causal / same length

    def _chomp(self, x):
        return x[:, :, :-self.chomp] if self.chomp > 0 else x

    def forward(self, x):
        out = self.relu(self._chomp(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self._chomp(self.conv2(out)))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class FeatureExtractor(nn.Module):
    """
    Shared temporal encoder f_theta : R^{T x D} -> R^{z}.

    Pipeline: 1D-CNN (local patterns) -> TCN blocks (multi-scale temporal
    dependencies, dilated) -> LSTM (sequential summary) -> linear projection
    to the latent embedding used for classification and OT alignment.
    """

    def __init__(self, num_channels: int, cnn_channels: int = 64,
                 tcn_channels: int = 64, lstm_hidden: int = 64,
                 feature_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(
            TemporalBlock(cnn_channels, tcn_channels, kernel_size=3, dilation=1, dropout=dropout),
            TemporalBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2, dropout=dropout),
        )
        self.lstm = nn.LSTM(input_size=tcn_channels, hidden_size=lstm_hidden,
                             num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(lstm_hidden * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) raw multivariate time-series batch.
        Returns:
            z: (B, feature_dim) latent embedding.
        """
        x = x.transpose(1, 2)          # (B, D, T) for Conv1d
        x = self.cnn(x)                # (B, cnn_channels, T)
        x = self.tcn(x)                # (B, tcn_channels, T)
        x = x.transpose(1, 2)          # (B, T, tcn_channels) for LSTM
        _, (h_n, _) = self.lstm(x)     # h_n: (2, B, lstm_hidden) [bidirectional]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 2*lstm_hidden)
        z = self.proj(h)               # (B, feature_dim)
        return z


class Classifier(nn.Module):
    """Classifier head h_phi : R^{z} -> Delta^K (softmax over classes)."""

    def __init__(self, feature_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, K); apply softmax externally where needed."""
        return self.net(z)
