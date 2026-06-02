"""Variational Autoencoder convolucional (CLAUDE.md §4.2).

Dos variantes con el MISMO interfaz, para que el resto de la app (muestreo del prior, mapa
latente, vecinos, interpolación, etc.) funcione con cualquiera de ellas:

  - "basico"  → `ConvVAE`      (base 32; la arquitectura original de la maqueta).
  - "grande"  → `BigConvVAE`   (base 64 + bloques residuales; más capacidad para reconstruir y generar).

Reutiliza el mismo stack convolucional que el AE determinista, pero el encoder produce dos
vectores —μ (media) y logσ² (log-varianza)— en vez de un único z. El latente se muestrea con el
*truco de reparametrización* `z = μ + σ·ε`, con `ε~N(0,I)`, de modo que el gradiente puede fluir a
través del muestreo.

Interfaz común (todas exponen):
  - `encode(x) -> (μ, logσ²)`        : 64×64×3 → dos vectores latentes (el posterior q(z|x)).
  - `reparameterize(μ, logσ²) -> z`  : z = μ + σ·ε con ε~N(0,I) (derivable).
  - `decode(z) -> img`               : z → 64×64×3 en [0,1].
  - `forward(x) -> (recon, μ, logσ², z)`.
  - `encode_mu(x) -> μ`              : latente determinista (la media) para mapa/vecinos/interpolación.
  - `sample(n, device) -> imgs`      : muestrea n caras del prior z~N(0,I) y decodifica.
  - `.latent_dim`                    : dimensión del latente.
  - `.arch`                          : nombre de la variante.

encoder: 64×64×3 → (conv ×4, /2 cada una) → 4×4×(BASE*8) → flatten → (fc_mu, fc_logvar)
decoder: z → fc → 4×4×(BASE*8) → (convT ×4, ×2 cada una) → 64×64×3, sigmoid → [0,1]
Pérdida: reconstrucción (MSE o L1) + β·KL.  KL contra el prior N(0,I).
"""

from __future__ import annotations

import torch
from torch import nn

from .ae import count_params  # noqa: F401  (re-exportado para el servicio)

IMG_CH = 3

# Variantes disponibles (en orden de presentación en la UI).
VAE_ARCHS = ("basico", "grande")


# ---------------------------------------------------------------------------
# Bloques reutilizables (espejo de los del AE)
# ---------------------------------------------------------------------------
def _down(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
    """Bloque de bajada: divide H,W entre 2 (stride 2)."""
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, 4, 2, 1)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up(cin: int, cout: int) -> nn.Sequential:
    """Bloque de subida: multiplica H,W por 2 (convtranspose stride 2)."""
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    """Bloque residual que conserva resolución y nº de canales (BatchNorm + LeakyReLU)."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


# ---------------------------------------------------------------------------
# Variante "basico" — la arquitectura original (base 32)
# ---------------------------------------------------------------------------
class ConvVAE(nn.Module):
    """VAE conv (base 32). encoder 64→4 (×4 conv /2) → (μ, logσ²); decoder simétrico."""

    arch = "basico"
    BASE = 32  # canales de la primera capa; se duplican hasta 256 (=BASE*8) en 4×4

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = self.BASE, self.BASE * 2, self.BASE * 4, self.BASE * 8  # 32,64,128,256

        # --- encoder: cada bloque divide H,W entre 2 (idéntico al AE) ---
        self.encoder = nn.Sequential(
            _down(IMG_CH, c1, norm=False),  # 64 -> 32
            _down(c1, c2),                  # 32 -> 16
            _down(c2, c3),                  # 16 -> 8
            _down(c3, c4),                  # 8  -> 4
        )
        self.enc_shape = (c4, 4, 4)
        flat = c4 * 4 * 4
        # dos cabezas lineales: media y log-varianza del posterior q(z|x)
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        # --- decoder: simétrico (idéntico al AE) ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            _up(c4, c3),                              # 4  -> 8
            _up(c3, c2),                              # 8  -> 16
            _up(c2, c1),                              # 16 -> 32
            nn.ConvTranspose2d(c1, IMG_CH, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
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


# ---------------------------------------------------------------------------
# Variante "grande" — base 64 + bloques residuales (más capacidad)
# ---------------------------------------------------------------------------
class BigConvVAE(nn.Module):
    """VAE conv de mayor capacidad: base 64 y bloques residuales en los niveles altos.

    Mismo latente probabilístico que el básico (μ, logσ² ∈ latent_dim), pero más canales y
    profundidad → reconstruye y genera mejor. Se omiten los ResBlock en el nivel más profundo
    (512×4×4): ahí son carísimos en parámetros y aportan poco, así el modelo se queda en el
    objetivo de ~6-10M. El interfaz es idéntico al de `ConvVAE`.
    """

    arch = "grande"
    BASE = 64  # 64,128,256,512 en 4×4

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = self.BASE, self.BASE * 2, self.BASE * 4, self.BASE * 8  # 64,128,256,512

        # --- encoder: bajada + ResBlock en cada nivel (salvo el cuello c4) ---
        self.encoder = nn.Sequential(
            _down(IMG_CH, c1, norm=False), ResBlock(c1),  # 64 -> 32
            _down(c1, c2), ResBlock(c2),                  # 32 -> 16
            _down(c2, c3), ResBlock(c3),                  # 16 -> 8
            _down(c3, c4),                                # 8  -> 4
        )
        self.enc_shape = (c4, 4, 4)
        flat = c4 * 4 * 4
        # dos cabezas lineales: media y log-varianza del posterior q(z|x)
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)

        # --- decoder: simétrico, ResBlock tras cada subida (salvo el cuello c4) ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            _up(c4, c3), ResBlock(c3),                # 4  -> 8
            _up(c3, c2), ResBlock(c2),                # 8  -> 16
            _up(c2, c1), ResBlock(c1),                # 16 -> 32
            nn.ConvTranspose2d(c1, IMG_CH, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
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
        """Latente determinista (la media μ), comparable al de la variante básica."""
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BUILDERS = {
    "basico": ConvVAE,
    "grande": BigConvVAE,
}


def build_vae(latent_dim: int, arch: str = "basico") -> nn.Module:
    """Construye la variante pedida. `arch` desconocida → 'basico' (degrada con elegancia)."""
    cls = _BUILDERS.get(arch, ConvVAE)
    return cls(latent_dim)
