"""Variational Autoencoder convolucional (CLAUDE.md §4.2).

Reutiliza el mismo stack convolucional que el AE determinista, pero el encoder
produce dos vectores —μ (media) y logσ² (log-varianza)— en vez de un único z.
El latente se muestrea con el *truco de reparametrización* `z = μ + σ·ε`, con
`ε~N(0,I)`, de modo que el gradiente puede fluir a través del muestreo.

encoder: 64×64×3 → (conv ×4, /2 cada una) → 4×4×256 → flatten → (fc_mu, fc_logvar)
decoder: z → fc → 4×4×256 → (convT ×4, ×2 cada una) → 64×64×3, sigmoid → [0,1]
Pérdida: reconstrucción (MSE o L1) + β·KL.  KL contra el prior N(0,I).
"""

from __future__ import annotations

import torch
from torch import nn

from .ae import count_params  # noqa: F401  (re-exportado para el servicio)

IMG_CH = 3
BASE = 32  # canales de la primera capa; se duplican hasta 256 (=BASE*8) en 4×4


class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = BASE, BASE * 2, BASE * 4, BASE * 8  # 32,64,128,256

        # --- encoder: cada bloque divide H,W entre 2 (idéntico al AE) ---
        self.encoder = nn.Sequential(
            self._down(IMG_CH, c1, norm=False),  # 64 -> 32
            self._down(c1, c2),                  # 32 -> 16
            self._down(c2, c3),                  # 16 -> 8
            self._down(c3, c4),                  # 8  -> 4
        )
        self.enc_shape = (c4, 4, 4)
        flat = c4 * 4 * 4
        # dos cabezas lineales: media y log-varianza del posterior q(z|x)
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        # --- decoder: simétrico (idéntico al AE) ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            self._up(c4, c3),                         # 4  -> 8
            self._up(c3, c2),                         # 8  -> 16
            self._up(c2, c1),                         # 16 -> 32
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

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x → (μ, logσ²) del posterior aproximado q(z|x)."""
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Truco de reparametrización: z = μ + σ·ε, con ε~N(0,I) y σ = exp(½·logσ²)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, *self.enc_shape)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        """Latente determinista (la media μ). Útil para el mapa latente, vecinos e
        interpolación: representa cada imagen por el centro de su posterior."""
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Genera n imágenes muestreando del prior z~N(0,I) y decodificando."""
        was_training = self.training
        self.eval()
        z = torch.randn(n, self.latent_dim, device=device)
        imgs = self.decode(z)
        if was_training:
            self.train()
        return imgs
