"""DCGAN para caras 64×64×3 (CLAUDE.md §4.3).

A diferencia del autoencoder, un GAN **no tiene encoder**: no comprime imágenes a un
código. Tiene dos redes que juegan una contra otra:

  Generator(z)      : z ∈ R^z_dim  → (ConvTranspose ×5) → imagen 64×64×3, tanh → [-1,1]
  Discriminator(x)  : imagen 64×64×3 → (Conv ×5)        → un logit (real vs. falso)

El generador parte de ruido z ~ N(0,1) y aprende a sintetizar caras; el discriminador
aprende a distinguir caras reales del dataset de las falsas del generador. Es la
arquitectura DCGAN clásica (Radford et al., 2016): BatchNorm + ReLU en el generador,
Conv stride-2 + LeakyReLU(0.2) en el discriminador, sin capas densas.

La salida del generador vive en [-1,1] (por la tanh). Para visualizar como PNG hay que
pasarla a [0,1] con `to_image_range`.
"""

from __future__ import annotations

import torch
from torch import nn

IMG_CH = 3
BASE = 64  # canales base; el generador parte de BASE*8 en 4×4 y el discriminador llega ahí


class Generator(nn.Module):
    """z (vector) → imagen 64×64×3 en [-1,1] mediante ConvTranspose."""

    def __init__(self, z_dim: int = 100) -> None:
        super().__init__()
        self.z_dim = z_dim
        c1, c2, c3, c4 = BASE * 8, BASE * 4, BASE * 2, BASE  # 512,256,128,64

        self.net = nn.Sequential(
            # z se trata como un "mapa" 1×1 de z_dim canales: (B, z_dim, 1, 1) → 4×4
            nn.ConvTranspose2d(z_dim, c1, 4, 1, 0, bias=False),  # 1  -> 4
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            self._up(c1, c2),                                    # 4  -> 8
            self._up(c2, c3),                                    # 8  -> 16
            self._up(c3, c4),                                    # 16 -> 32
            nn.ConvTranspose2d(c4, IMG_CH, 4, 2, 1, bias=False),  # 32 -> 64
            nn.Tanh(),                                            # salida en [-1,1]
        )

    @staticmethod
    def _up(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # acepta z como (B, z_dim) o (B, z_dim, 1, 1)
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """imagen 64×64×3 → un logit (sin sigmoid; la BCE se hace con logits)."""

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = BASE, BASE * 2, BASE * 4, BASE * 8  # 64,128,256,512

        self.net = nn.Sequential(
            self._down(IMG_CH, c1, norm=False),  # 64 -> 32  (sin norm en la 1ª, como DCGAN)
            self._down(c1, c2),                  # 32 -> 16
            self._down(c2, c3),                  # 16 -> 8
            self._down(c3, c4),                  # 8  -> 4
            nn.Conv2d(c4, 1, 4, 1, 0, bias=False),  # 4  -> 1  (logit)
        )

    @staticmethod
    def _down(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(cout))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)  # (B,) logits


def init_weights(module: nn.Module) -> None:
    """Inicialización DCGAN: pesos N(0, 0.02); BatchNorm γ~N(1,0.02), β=0."""
    name = module.__class__.__name__
    if "Conv" in name:
        if module.weight is not None:  # type: ignore[attr-defined]
            nn.init.normal_(module.weight.data, 0.0, 0.02)  # type: ignore[attr-defined]
    elif "BatchNorm" in name:
        nn.init.normal_(module.weight.data, 1.0, 0.02)  # type: ignore[attr-defined]
        nn.init.constant_(module.bias.data, 0.0)        # type: ignore[attr-defined]


def to_image_range(x: torch.Tensor) -> torch.Tensor:
    """Pasa la salida del generador de [-1,1] a [0,1] para visualizar (CHW float)."""
    return (x.clamp(-1, 1) + 1) / 2


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
