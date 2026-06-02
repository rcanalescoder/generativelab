"""Autoencoders convolucionales deterministas (CLAUDE.md §4.1).

Tres variantes con el MISMO interfaz para que el resto de la app (mapa latente, vecinos,
interpolación, etc.) funcione con cualquiera de ellas:

  - "basico"  → `ConvAutoencoder`   (base 32; la arquitectura original de la maqueta).
  - "grande"  → `BigAutoencoder`    (base 64 + bloques residuales; más capacidad).
  - "unet"    → `UNetAutoencoder`   (encoder con skip connections; reconstrucción nítida).

Interfaz común (todas exponen):
  - `encode(x) -> z`            : 64×64×3 → vector latente (el "cuello de botella").
  - `decode(z) -> img`          : z → 64×64×3 en [0,1] (sin skips; rellena con ceros en U-Net).
  - `forward(x) -> (recon, z)`  : recon usa skips si la variante los tiene (→ más nítido).
  - `.latent_dim`               : dimensión del latente.
  - `.arch`                     : nombre de la variante.

Detalle pedagógico de la U-Net: `encode(x)` devuelve SOLO el cuello (sin skips), de modo que el
mapa latente / vecinos / interpolación operan sobre un latente comparable al de las otras
variantes. `forward(x)` sí usa los skips y reconstruye nítido. Cuando decodificamos desde un z
suelto (interpolación, ruido en z) NO hay skips → la imagen sale más borrosa: justamente la
lección de que los skips "puentean" el cuello.

Pérdida: MSE o L1 (se elige en el entrenamiento).
"""

from __future__ import annotations

import torch
from torch import nn

IMG_CH = 3

# Variantes disponibles (en orden de presentación en la UI).
ARCHS = ("basico", "grande", "unet")


# ---------------------------------------------------------------------------
# Bloques reutilizables
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
class ConvAutoencoder(nn.Module):
    """AE conv determinista (base 32). encoder 64→4 (×4 conv /2), decoder simétrico."""

    arch = "basico"
    BASE = 32  # canales de la primera capa; se duplican hasta 256 (=BASE*8) en 4×4

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = self.BASE, self.BASE * 2, self.BASE * 4, self.BASE * 8  # 32,64,128,256

        # --- encoder: cada bloque divide H,W entre 2 ---
        self.encoder = nn.Sequential(
            _down(IMG_CH, c1, norm=False),  # 64 -> 32
            _down(c1, c2),                  # 32 -> 16
            _down(c2, c3),                  # 16 -> 8
            _down(c3, c4),                  # 8  -> 4
        )
        self.enc_shape = (c4, 4, 4)
        flat = c4 * 4 * 4
        self.fc_enc = nn.Linear(flat, latent_dim)

        # --- decoder: simétrico ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            _up(c4, c3),                              # 4  -> 8
            _up(c3, c2),                              # 8  -> 16
            _up(c2, c1),                              # 16 -> 32
            nn.ConvTranspose2d(c1, IMG_CH, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
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


# ---------------------------------------------------------------------------
# Variante "grande" — base 64 + bloques residuales (más capacidad)
# ---------------------------------------------------------------------------
class BigAutoencoder(nn.Module):
    """AE conv de mayor capacidad: base 64 y bloques residuales en los niveles altos.

    Mismo cuello que el básico (z = latent_dim), pero más canales y profundidad → reconstruye
    mejor. Se omiten los ResBlock en el nivel más profundo (512×4×4): ahí son carísimos en
    parámetros y aportan poco, así el modelo se queda en el objetivo de ~6-12M.
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
        self.fc_enc = nn.Linear(flat, latent_dim)

        # --- decoder: simétrico, ResBlock tras cada subida (salvo el cuello c4) ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            _up(c4, c3), ResBlock(c3),                # 4  -> 8
            _up(c3, c2), ResBlock(c2),                # 8  -> 16
            _up(c2, c1), ResBlock(c1),                # 16 -> 32
            nn.ConvTranspose2d(c1, IMG_CH, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
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


# ---------------------------------------------------------------------------
# Variante "unet" — encoder con skip connections
# ---------------------------------------------------------------------------
class UNetAutoencoder(nn.Module):
    """AE con skip connections estilo U-Net (base 48).

    - `forward(x)`: usa los skips del encoder al decodificar → reconstrucción nítida.
    - `encode(x)`: devuelve SOLO el cuello (vector z), SIN skips, para que el mapa latente /
      vecinos / interpolación sean comparables a las otras variantes.
    - `decode(z, skips=None)`: si no hay skips, los rellena con ceros → reconstrucción más
      borrosa desde z (lección: los skips puentean el cuello de botella).
    """

    arch = "unet"
    BASE = 48  # 48,96,192,384 en 4×4

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        c1, c2, c3, c4 = self.BASE, self.BASE * 2, self.BASE * 4, self.BASE * 8  # 48,96,192,384

        # --- encoder por niveles (guardamos cada salida como skip) ---
        self.enc1 = _down(IMG_CH, c1, norm=False)  # 64 -> 32   (skip s1: c1×32×32)
        self.enc2 = _down(c1, c2)                  # 32 -> 16   (skip s2: c2×16×16)
        self.enc3 = _down(c2, c3)                  # 16 -> 8    (skip s3: c3×8×8)
        self.enc4 = _down(c3, c4)                  # 8  -> 4    (bottleneck feat: c4×4×4)
        self._skip_ch = (c1, c2, c3)
        self._skip_hw = (32, 16, 8)

        self.enc_shape = (c4, 4, 4)
        flat = c4 * 4 * 4
        self.fc_enc = nn.Linear(flat, latent_dim)

        # --- decoder: en cada nivel concatena el skip correspondiente ---
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.dec3 = _up(c4, c3)            # 4  -> 8   (luego concat s3 → 2*c3)
        self.dec2 = _up(c3 * 2, c2)        # 8  -> 16  (luego concat s2 → 2*c2)
        self.dec1 = _up(c2 * 2, c1)        # 16 -> 32  (luego concat s1 → 2*c1)
        self.out = nn.Sequential(
            nn.ConvTranspose2d(c1 * 2, IMG_CH, 4, 2, 1),  # 32 -> 64
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Solo el cuello (sin skips), para el espacio latente comparable."""
        h = self.enc4(self.enc3(self.enc2(self.enc1(x))))
        return self.fc_enc(h.flatten(1))

    def encode_with_skips(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Devuelve (z, [s1, s2, s3]) para la reconstrucción nítida del forward."""
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        h = self.enc4(s3)
        z = self.fc_enc(h.flatten(1))
        return z, [s1, s2, s3]

    def decode(self, z: torch.Tensor, skips: list[torch.Tensor] | None = None) -> torch.Tensor:
        """Decodifica z. Sin skips → se rellenan con ceros (reconstrucción más borrosa)."""
        h = self.fc_dec(z).view(-1, *self.enc_shape)
        b = z.shape[0]
        if skips is None:
            skips = [
                torch.zeros(b, ch, hw, hw, device=z.device, dtype=z.dtype)
                for ch, hw in zip(self._skip_ch, self._skip_hw)
            ]
        s1, s2, s3 = skips
        h = torch.cat([self.dec3(h), s3], dim=1)  # 8×8
        h = torch.cat([self.dec2(h), s2], dim=1)  # 16×16
        h = torch.cat([self.dec1(h), s1], dim=1)  # 32×32
        return self.out(h)                        # 64×64

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z, skips = self.encode_with_skips(x)
        return self.decode(z, skips), z


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BUILDERS = {
    "basico": ConvAutoencoder,
    "grande": BigAutoencoder,
    "unet": UNetAutoencoder,
}


def build_autoencoder(latent_dim: int, arch: str = "basico") -> nn.Module:
    """Construye la variante pedida. `arch` desconocida → 'basico' (degrada con elegancia)."""
    cls = _BUILDERS.get(arch, ConvAutoencoder)
    return cls(latent_dim)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
