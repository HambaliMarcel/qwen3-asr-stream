"""Minimal PANNs CNN6 (AudioSet tagging). Vendored from audioset_tagging_cnn."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram


def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if getattr(layer, "bias", None) is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input: torch.Tensor, pool_size=(2, 2), pool_type: str = "avg") -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(input)))
        if pool_type == "max":
            return F.max_pool2d(x, kernel_size=pool_size)
        if pool_type == "avg":
            return F.avg_pool2d(x, kernel_size=pool_size)
        raise ValueError(f"Unknown pool_type: {pool_type}")


class Cnn6(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        window_size: int,
        hop_size: int,
        mel_bins: int,
        fmin: int,
        fmax: int,
        classes_num: int,
    ) -> None:
        super().__init__()
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=1.0,
            amin=1e-10,
            top_db=None,
            freeze_parameters=True,
        )
        self.bn0 = nn.BatchNorm2d(64)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input: torch.Tensor, mixup_lambda=None) -> dict[str, torch.Tensor]:
        x = self.spectrogram_extractor(input)
        x = self.logmel_extractor(x)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        clipwise_output = torch.sigmoid(self.fc_audioset(x))
        return {"clipwise_output": clipwise_output, "embedding": embedding}
