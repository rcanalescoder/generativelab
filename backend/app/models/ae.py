"""Autoencoder convolucional determinista (CLAUDE.md §4.1).

encoder: 64×64×3 → (conv ×4, /2 cada una) → 4×4×256 → fc → z(latent_dim)
decoder: z → fc → 4×4×256 → (convT ×4, ×2 cada una) → 64×64×3, sigmoid → [0,1]
Pérdida: MSE o L1 (se elige en el entrenamiento).
"""

from __future__ import annotations

import torch
from torch import nn

IMG_CH = 3
BASE = 32  # canales de la primera capa; se duplican hasta 256 (=BASE*8) en 4×4


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = BASE, BASE * 2, BASE * 4, BASE * 8  # 32,64,128,256

        # --- encoder: cada bloque divide H,W entre 2 ---
        self.encoder = nn.Sequential(
            self._down(IMG_CH, c1, norm=False),  # 64 -> 32
            self._down(c1, c2),                  # 32 -> 16
            self._down(c2, c3),                  # 16 -> 8
            self._down(c3, c4),                  # 8  -> 4
        )
        self.enc_shape = (c4, 4, 4)
        flat = c4 * 4 * 4
        self.fc_enc = nn.Linear(flat, latent_dim)

        # --- decoder: simétrico ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            self._up(c4, c3),                       # 4  -> 8
            self._up(c3, c2),                       # 8  -> 16
            self._up(c2, c1),                       # 16 -> 32
            nn.ConvTranspose2d(c1, IMG_CH, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
        )

    @staticmethod
    def _down(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, 4, 2, 1)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    @staticmethod
    def _up(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, 2, 1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.fc_enc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, *self.enc_shape)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
