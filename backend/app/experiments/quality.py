"""Métricas de calidad de GENERACIÓN para el ciclo v3 («Lista de Mejoras.md»).

La v2 solo medía reconstrucción (MSE/PSNR/SSIM): un modelo colapsado puntúa «bien» ahí
mientras genera manchas. Este módulo mide la GENERACIÓN comparando la distribución de las
muestras generadas con la de caras reales:

  - KID (Kernel Inception Distance, Bińkowski et al. 2018): MMD² insesgada con kernel
    polinómico sobre features de InceptionV3 (2048-d). Se prefiere a FID porque su estimador
    es insesgado con pocas muestras (n≈2k), lo único asumible en local/MPS. Menor = mejor.
  - Diversidad: distancia media por pares entre features de las muestras, y su ratio frente
    a la del dataset real (ratio→0 = colapso de modo; ≈1 = tan variada como el dataset).
  - Rejilla 8×8 con semilla fija (save_grid) → comparación visual directa entre runs.

Las features del held-out real se cachean en experiments/cache/ (npz). Todo el protocolo es
determinista: mismas semillas ⇒ mismos números.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..data import dataset
from ..device import get_device

# Artefactos del ciclo v3 (repo-raíz /experiments): cache de features, grids, ledger.
EXP_DIR = Path(__file__).resolve().parents[3] / "experiments"
CACHE_DIR = EXP_DIR / "cache"
GRIDS_DIR = EXP_DIR / "grids"
LEDGER_PATH = EXP_DIR / "ledger.jsonl"

REAL_SEED = 1234        # semilla del held-out real (independiente de los entrenamientos)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

_INCEPTION: nn.Module | None = None


def _inception() -> nn.Module:
    """InceptionV3 preentrenada (ImageNet) con la cabeza fc sustituida por identidad.

    Devuelve features de 2048-d (pool final). Singleton: se construye una vez por proceso.
    """
    global _INCEPTION
    if _INCEPTION is None:
        from torchvision.models import Inception_V3_Weights, inception_v3

        net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        net.eval()
        _INCEPTION = net.to(get_device())
    return _INCEPTION


@torch.no_grad()
def features(imgs01: torch.Tensor, batch_size: int = 64) -> np.ndarray:
    """Features InceptionV3 (N,2048) de imágenes NCHW en [0,1] (cualquier resolución)."""
    net = _inception()
    dev = next(net.parameters()).device
    mean = torch.tensor(_IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    out: list[np.ndarray] = []
    for i in range(0, imgs01.shape[0], batch_size):
        x = imgs01[i : i + batch_size].to(dev).float().clamp(0, 1)
        x = F.interpolate(x, size=299, mode="bilinear", align_corners=False)
        x = (x - mean) / std
        out.append(net(x).cpu().numpy())
    return np.concatenate(out, 0).astype(np.float64)


def real_features(n: int = 2048, batch_size: int = 64) -> np.ndarray:
    """Features del held-out REAL fijo (seed REAL_SEED). Se cachean en disco (npz)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"real_inception_n{n}_seed{REAL_SEED}.npz"
    if path.exists():
        return np.load(path)["feats"]
    total = dataset.count()
    rng = np.random.default_rng(REAL_SEED)
    ids = np.sort(rng.choice(total, size=min(n, total), replace=False))
    arr = dataset.load_array()
    chunks: list[torch.Tensor] = []
    for i in range(0, len(ids), 256):
        a = np.asarray(arr[ids[i : i + 256]])
        chunks.append(torch.from_numpy(a).float().div(255).permute(0, 3, 1, 2))
    feats = features(torch.cat(chunks, 0), batch_size=batch_size)
    np.savez_compressed(path, feats=feats, ids=ids)
    return feats


def _poly_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Kernel polinómico estándar del KID: k(x,y) = (x·y/d + 1)³."""
    d = x.shape[1]
    return (x @ y.T / d + 1.0) ** 3


def kid(
    real: np.ndarray, fake: np.ndarray, subsets: int = 10, subset_size: int = 1024, seed: int = 0
) -> tuple[float, float]:
    """KID×1000 (media, desviación) por subconjuntos, con MMD² insesgada.

    Para cada subconjunto se muestrean `m` features reales y `m` falsas sin reemplazo y se
    calcula la MMD² insesgada (sin la diagonal en los términos intra-conjunto).
    """
    m = int(min(subset_size, real.shape[0], fake.shape[0]))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(subsets):
        xr = real[rng.choice(real.shape[0], m, replace=False)]
        xf = fake[rng.choice(fake.shape[0], m, replace=False)]
        k_rr = _poly_kernel(xr, xr)
        k_ff = _poly_kernel(xf, xf)
        k_rf = _poly_kernel(xr, xf)
        mmd = (
            (k_rr.sum() - np.trace(k_rr)) / (m * (m - 1))
            + (k_ff.sum() - np.trace(k_ff)) / (m * (m - 1))
            - 2.0 * k_rf.mean()
        )
        vals.append(float(mmd))
    v = np.asarray(vals)
    return float(v.mean() * 1000.0), float(v.std() * 1000.0)


def diversity(feats: np.ndarray, max_n: int = 1024, seed: int = 0) -> float:
    """Distancia euclídea media por pares entre features (proxy de diversidad/colapso)."""
    rng = np.random.default_rng(seed)
    n = int(min(max_n, feats.shape[0]))
    x = feats[rng.choice(feats.shape[0], n, replace=False)]
    sq = (x * x).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (x @ x.T), 0.0)
    iu = np.triu_indices(n, 1)
    return float(np.sqrt(d2[iu]).mean())


def save_grid(imgs01: torch.Tensor, path: Path, ncol: int = 8, upscale: int = 2) -> Path:
    """Guarda una rejilla PNG (filas×ncol) de imágenes NCHW [0,1], reescaladas ×upscale."""
    from PIL import Image

    imgs = imgs01.detach().cpu().clamp(0, 1)
    n = imgs.shape[0]
    ncol = min(ncol, n)
    nrow = (n + ncol - 1) // ncol
    h, w = imgs.shape[2] * upscale, imgs.shape[3] * upscale
    pad = 2
    canvas = Image.new("RGB", (ncol * w + (ncol + 1) * pad, nrow * h + (nrow + 1) * pad), (245, 246, 249))
    for k in range(n):
        a = (imgs[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        im = Image.fromarray(a).resize((w, h), Image.NEAREST)
        r, c = divmod(k, ncol)
        canvas.paste(im, (pad + c * (w + pad), pad + r * (h + pad)))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def evaluate_generation(
    fake01: torch.Tensor,
    *,
    real_n: int = 2048,
    subset_size: int | None = None,
    grid_path: Path | None = None,
    grid_n: int = 64,
) -> dict:
    """Protocolo completo: KID + diversidad (+grid) de un lote de muestras generadas [0,1]."""
    rf = real_features(real_n)
    ff = features(fake01)
    m = subset_size or int(min(1024, ff.shape[0], rf.shape[0]))
    kid_mean, kid_std = kid(rf, ff, subset_size=m)
    div_fake = diversity(ff)
    div_real = diversity(rf)
    out = {
        "n_fake": int(ff.shape[0]),
        "n_real": int(rf.shape[0]),
        "kid_subset": m,
        "kid_x1000": round(kid_mean, 2),
        "kid_std_x1000": round(kid_std, 2),
        "diversity": round(div_fake, 3),
        "diversity_real": round(div_real, 3),
        "diversity_ratio": round(div_fake / max(div_real, 1e-9), 3),
    }
    if grid_path is not None:
        save_grid(fake01[:grid_n], grid_path)
        out["grid"] = str(grid_path)
    return out
