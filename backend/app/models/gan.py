"""DCGAN para caras 64×64×3 (CLAUDE.md §4.3), con dos variantes de capacidad.

A diferencia del autoencoder, un GAN **no tiene encoder**: no comprime imágenes a un
código. Tiene dos redes que juegan una contra otra:

  Generator(z)      : z ∈ R^z_dim  → (ConvTranspose) → imagen 64×64×3, tanh → [-1,1]
  Discriminator(x)  : imagen 64×64×3 → (Conv)        → un logit (real vs. falso)

El generador parte de ruido z ~ N(0,1) y aprende a sintetizar caras; el discriminador
aprende a distinguir caras reales del dataset de las falsas del generador.

Dos variantes con el MISMO interfaz (igual que el AE), para poder elegir y comparar la
calidad de las caras generadas (la comparación del GAN es **visual**, no por métrica de
reconstrucción):

  - "basico"  → DCGAN clásico (Radford et al., 2016), base 64. La maqueta original.
  - "grande"  → más canales base (96) y una capa extra de refinado a resolución plena en
                G y D → más capacidad (caras más nítidas/detalladas). G+D ≈ 14M parámetros.

Interfaz común (ambas variantes):
  - `Generator.forward(z) -> img`  : z (B,z_dim) o (B,z_dim,1,1) → imagen 64×64×3 en [-1,1].
  - `Discriminator.forward(img) -> logit` : imagen → (B,) logits (sin sigmoid; BCE con logits).
  - `Generator.z_dim`              : dimensión del ruido de entrada.
  - `Generator.arch` / `Discriminator.arch` : nombre de la variante.

La salida del generador vive en [-1,1] (por la tanh). Para visualizar como PNG hay que
pasarla a [0,1] con `to_image_range`.
"""

from __future__ import annotations

import torch
from torch import nn

IMG_CH = 3

# Variantes disponibles (en orden de presentación en la UI).
GAN_ARCHS = ("basico", "grande")


# ---------------------------------------------------------------------------
# Bloques reutilizables
# ---------------------------------------------------------------------------
def _g_up(cin: int, cout: int) -> nn.Sequential:
    """Bloque de subida del generador: duplica H,W (ConvTranspose stride-2) + BN + ReLU."""
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


def _g_refine(ch: int) -> nn.Sequential:
    """Bloque de refinado del generador: conserva resolución (Conv 3×3 stride-1) + BN + ReLU."""
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(ch),
        nn.ReLU(inplace=True),
    )


def _d_down(cin: int, cout: int, norm: bool = True) -> nn.Sequential:
    """Bloque de bajada del discriminador: divide H,W entre 2 (Conv stride-2) + LeakyReLU."""
    layers: list[nn.Module] = [nn.Conv2d(cin, cout, 4, 2, 1, bias=False)]
    if norm:
        layers.append(nn.BatchNorm2d(cout))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _d_refine(ch: int) -> nn.Sequential:
    """Bloque de refinado del discriminador: conserva resolución (Conv 3×3 stride-1) + LeakyReLU."""
    return nn.Sequential(
        nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


# ---------------------------------------------------------------------------
# Variante "basico" — el DCGAN clásico (base 64)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """z (vector) → imagen 64×64×3 en [-1,1] mediante ConvTranspose (DCGAN, base 64)."""

    arch = "basico"
    BASE = 64  # el generador parte de BASE*8 en 4×4

    def __init__(self, z_dim: int = 100) -> None:
        super().__init__()
        self.z_dim = z_dim
        c1, c2, c3, c4 = self.BASE * 8, self.BASE * 4, self.BASE * 2, self.BASE  # 512,256,128,64

        self.net = nn.Sequential(
            # z se trata como un "mapa" 1×1 de z_dim canales: (B, z_dim, 1, 1) → 4×4
            nn.ConvTranspose2d(z_dim, c1, 4, 1, 0, bias=False),  # 1  -> 4
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            _g_up(c1, c2),                                       # 4  -> 8
            _g_up(c2, c3),                                       # 8  -> 16
            _g_up(c3, c4),                                       # 16 -> 32
            nn.ConvTranspose2d(c4, IMG_CH, 4, 2, 1, bias=False),  # 32 -> 64
            nn.Tanh(),                                            # salida en [-1,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # acepta z como (B, z_dim) o (B, z_dim, 1, 1)
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class Discriminator(nn.Module):
    """imagen 64×64×3 → un logit (sin sigmoid; la BCE se hace con logits). Base 64."""

    arch = "basico"
    BASE = 64

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = self.BASE, self.BASE * 2, self.BASE * 4, self.BASE * 8  # 64,128,256,512

        self.net = nn.Sequential(
            _d_down(IMG_CH, c1, norm=False),  # 64 -> 32  (sin norm en la 1ª, como DCGAN)
            _d_down(c1, c2),                  # 32 -> 16
            _d_down(c2, c3),                  # 16 -> 8
            _d_down(c3, c4),                  # 8  -> 4
            nn.Conv2d(c4, 1, 4, 1, 0, bias=False),  # 4  -> 1  (logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)  # (B,) logits


# ---------------------------------------------------------------------------
# Variante "grande" — más canales (base 96) + una capa extra de refinado
# ---------------------------------------------------------------------------
class BigGenerator(nn.Module):
    """Generador de mayor capacidad: base 96 y un bloque de refinado a resolución plena.

    Mismo esqueleto que el básico (z → 4×4 → … → 64×64) pero con más canales y, antes de la
    capa de salida, una ConvTranspose 32→64 seguida de un Conv 3×3 stride-1 que "pule" la
    imagen a resolución completa. Más capacidad → caras más nítidas/detalladas.
    """

    arch = "grande"
    BASE = 96  # el generador parte de BASE*8 en 4×4

    def __init__(self, z_dim: int = 100) -> None:
        super().__init__()
        self.z_dim = z_dim
        c1, c2, c3, c4 = self.BASE * 8, self.BASE * 4, self.BASE * 2, self.BASE  # 768,384,192,96

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, c1, 4, 1, 0, bias=False),  # 1  -> 4
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            _g_up(c1, c2),                                       # 4  -> 8
            _g_up(c2, c3),                                       # 8  -> 16
            _g_up(c3, c4),                                       # 16 -> 32
            _g_up(c4, c4),                                       # 32 -> 64
            _g_refine(c4),                                       # 64×64 (capa extra de refinado)
            nn.Conv2d(c4, IMG_CH, 3, 1, 1, bias=False),         # 64×64 -> imagen
            nn.Tanh(),                                            # salida en [-1,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)


class BigDiscriminator(nn.Module):
    """Discriminador de mayor capacidad: base 96 + un bloque de refinado a 32×32.

    Espejo del `BigGenerator`: tras la primera bajada (64→32) añade un Conv 3×3 stride-1 que
    aporta una capa extra de proceso antes de seguir reduciendo la resolución.
    """

    arch = "grande"
    BASE = 96

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3, c4 = self.BASE, self.BASE * 2, self.BASE * 4, self.BASE * 8  # 96,192,384,768

        self.net = nn.Sequential(
            _d_down(IMG_CH, c1, norm=False),  # 64 -> 32  (sin norm en la 1ª, como DCGAN)
            _d_refine(c1),                    # 32×32 (capa extra de refinado)
            _d_down(c1, c2),                  # 32 -> 16
            _d_down(c2, c3),                  # 16 -> 8
            _d_down(c3, c4),                  # 8  -> 4
            nn.Conv2d(c4, 1, 4, 1, 0, bias=False),  # 4  -> 1  (logit)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)  # (B,) logits


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BUILDERS: dict[str, tuple[type[nn.Module], type[nn.Module]]] = {
    "basico": (Generator, Discriminator),
    "grande": (BigGenerator, BigDiscriminator),
}


def build_gan(z_dim: int, arch: str = "basico") -> tuple[nn.Module, nn.Module]:
    """Construye (Generator, Discriminator) de la variante pedida.

    `arch` desconocida → 'basico' (degrada con elegancia, igual que `build_autoencoder`).
    """
    g_cls, d_cls = _BUILDERS.get(arch, _BUILDERS["basico"])
    return g_cls(z_dim), d_cls()


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


# ---------------------------------------------------------------------------
# DiffAugment (Zhao et al., 2020) — mejora v3, ver «Lista de Mejoras.md» §4
# ---------------------------------------------------------------------------
# Augmentación DIFERENCIABLE aplicada a TODO lo que ve el discriminador (reales y falsas,
# también en el paso del generador). Como real y falso se transforman igual, el equilibrio
# del juego no cambia, pero el discriminador ya no puede memorizar imágenes concretas →
# menos sobreajuste de D y entrenamiento mucho más estable con datasets de decenas de miles
# de imágenes. Es diferenciable, así que el gradiente fluye hasta G sin trucos extra.

def _rand_brightness(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _rand_saturation(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean


def _rand_contrast(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean


def _rand_translation(x: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    """Desplaza cada imagen hasta ±ratio·tamaño (rellena con 0), por muestra del lote."""
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = nn.functional.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    return x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()


def _rand_cutout(x: torch.Tensor, ratio: float = 0.5) -> torch.Tensor:
    """Tapa un cuadrado aleatorio (½ del tamaño) por muestra: D no puede fijarse en una zona."""
    cut_h = int(x.size(2) * ratio + 0.5)
    cut_w = int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cut_h % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cut_w % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), device=x.device),
        torch.arange(cut_h, device=x.device),
        torch.arange(cut_w, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(grid_x + offset_x - cut_h // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cut_w // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


_DIFFAUG_FNS = {
    "color": (_rand_brightness, _rand_saturation, _rand_contrast),
    "translation": (_rand_translation,),
    "cutout": (_rand_cutout,),
}


def diff_augment(x: torch.Tensor, policy: str = "color,translation,cutout") -> torch.Tensor:
    """Aplica la política DiffAugment a un lote en [-1,1]. Diferenciable (sin .detach())."""
    for name in policy.split(","):
        for fn in _DIFFAUG_FNS[name.strip()]:
            x = fn(x)
    return x.contiguous()
