"""FBCNN architecture for ONNX conversion.

Reconstructed from jiaxi-jiang/FBCNN (Apache 2.0 license).
This file defines the network architecture needed to load
fbcnn_color.pth and export to ONNX.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block: Conv-ReLU-Conv with skip connection."""

    def __init__(self, nc=64):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
        )

    def forward(self, x):
        return x + self.res(x)


class QFAttention(nn.Module):
    """Quality-factor-modulated residual block."""

    def __init__(self, nc=64):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
        )

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        res = gamma * self.res(x) + beta
        return x + res


class FBCNN(nn.Module):
    """Flexible Blind CNN for JPEG artifact removal.

    Architecture: U-Net with QF prediction and attention.
    """

    def __init__(self, in_nc=3, out_nc=3, nc=None, nb=4):
        super().__init__()
        if nc is None:
            nc = [64, 128, 256, 512]
        self.nb = nb

        # Head
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1)

        # Encoder
        self.m_down1 = nn.Sequential(
            *[ResBlock(nc[0]) for _ in range(nb)],
            nn.Conv2d(nc[0], nc[1], 2, 2),
        )
        self.m_down2 = nn.Sequential(
            *[ResBlock(nc[1]) for _ in range(nb)],
            nn.Conv2d(nc[1], nc[2], 2, 2),
        )
        self.m_down3 = nn.Sequential(
            *[ResBlock(nc[2]) for _ in range(nb)],
            nn.Conv2d(nc[2], nc[3], 2, 2),
        )

        # Bottleneck
        self.m_body_encoder = nn.Sequential(
            *[ResBlock(nc[3]) for _ in range(nb)]
        )
        self.m_body_decoder = nn.Sequential(
            *[ResBlock(nc[3]) for _ in range(nb)]
        )

        # QF prediction
        self.qf_pred = nn.Sequential(
            *[ResBlock(nc[3]) for _ in range(nb)],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(nc[3], nc[3]),
            nn.ReLU(inplace=True),
            nn.Linear(nc[3], nc[3]),
            nn.ReLU(inplace=True),
            nn.Linear(nc[3], 1),
            nn.Sigmoid(),
        )

        # QF embedding
        self.qf_embed = nn.Sequential(
            nn.Linear(1, nc[3]),
            nn.ReLU(inplace=True),
            nn.Linear(nc[3], nc[3]),
            nn.ReLU(inplace=True),
            nn.Linear(nc[3], nc[3]),
            nn.ReLU(inplace=True),
        )

        # Gamma/beta projections for QFAttention
        self.to_gamma_3 = nn.Sequential(
            nn.Linear(nc[3], nc[2]), nn.Sigmoid()
        )
        self.to_beta_3 = nn.Sequential(
            nn.Linear(nc[3], nc[2]), nn.Tanh()
        )
        self.to_gamma_2 = nn.Sequential(
            nn.Linear(nc[3], nc[1]), nn.Sigmoid()
        )
        self.to_beta_2 = nn.Sequential(
            nn.Linear(nc[3], nc[1]), nn.Tanh()
        )
        self.to_gamma_1 = nn.Sequential(
            nn.Linear(nc[3], nc[0]), nn.Sigmoid()
        )
        self.to_beta_1 = nn.Sequential(
            nn.Linear(nc[3], nc[0]), nn.Tanh()
        )

        # Decoder: ConvTranspose + QFAttention blocks
        # The checkpoint stores QFAttention blocks as m_up*.1, .2, etc
        self.m_up3 = nn.ModuleList([
            nn.ConvTranspose2d(nc[3], nc[2], 2, 2),
            *[QFAttention(nc[2]) for _ in range(nb)],
        ])

        self.m_up2 = nn.ModuleList([
            nn.ConvTranspose2d(nc[2], nc[1], 2, 2),
            *[QFAttention(nc[1]) for _ in range(nb)],
        ])

        self.m_up1 = nn.ModuleList([
            nn.ConvTranspose2d(nc[1], nc[0], 2, 2),
            *[QFAttention(nc[0]) for _ in range(nb)],
        ])

        # Tail
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1)

    def forward(self, x):
        # Pad to multiple of 8
        h, w = x.shape[2], x.shape[3]
        ph = (8 - h % 8) % 8
        pw = (8 - w % 8) % 8
        x = nn.functional.pad(x, (0, pw, 0, ph), mode='replicate')

        # Encoder
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)

        # Bottleneck
        x = self.m_body_encoder(x4)
        qf = self.qf_pred(x)
        x = self.m_body_decoder(x)

        # QF embedding
        qf_emb = self.qf_embed(qf)
        gamma_3 = self.to_gamma_3(qf_emb)
        beta_3 = self.to_beta_3(qf_emb)
        gamma_2 = self.to_gamma_2(qf_emb)
        beta_2 = self.to_beta_2(qf_emb)
        gamma_1 = self.to_gamma_1(qf_emb)
        beta_1 = self.to_beta_1(qf_emb)

        # Decoder - skip connections match original FBCNN
        x = x + x4
        x = self.m_up3[0](x)
        for i in range(len(self.m_up3) - 1):
            x = self.m_up3[i + 1](x, gamma_3, beta_3)

        x = x + x3
        x = self.m_up2[0](x)
        for i in range(len(self.m_up2) - 1):
            x = self.m_up2[i + 1](x, gamma_2, beta_2)

        x = x + x2
        x = self.m_up1[0](x)
        for i in range(len(self.m_up1) - 1):
            x = self.m_up1[i + 1](x, gamma_1, beta_1)

        x = x + x1
        x = self.m_tail(x)

        # Crop back to original size
        return x[:, :, :h, :w]
