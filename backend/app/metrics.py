"""Métricas de reconstrucción de imágenes: MSE, PSNR y SSIM.

Funciones puras en PyTorch, sin dependencias nuevas. Trabajan sobre imágenes en [0,1] con
forma CHW `(C,H,W)` o por lotes NCHW `(N,C,H,W)`. Por lote devuelven la media sobre N.

- `mse`  : error cuadrático medio por píxel.
- `psnr` : Peak Signal-to-Noise Ratio en dB (mayor = mejor; ∞ si idénticas).
- `ssim` : Structural Similarity Index con ventana gaussiana 11×11 (implementado a mano),
           promediado sobre canales. Rango [-1,1]; ~1 = idénticas.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-12


def _as_batch(t: torch.Tensor) -> torch.Tensor:
    """Garantiza forma NCHW (añade dim de lote si viene CHW)."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 4:
        return t
    raise ValueError(f"se esperaba CHW o NCHW, recibido {tuple(t.shape)}")


def mse(x: torch.Tensor, xhat: torch.Tensor) -> float:
    """MSE por píxel, promediado sobre el lote. Imágenes en [0,1]."""
    a = _as_batch(x).float()
    b = _as_batch(xhat).float()
    per_img = torch.mean((a - b) ** 2, dim=[1, 2, 3])
    return float(per_img.mean().item())


def psnr(x: torch.Tensor, xhat: torch.Tensor, max_val: float = 1.0) -> float:
    """PSNR medio en dB (rango de datos `max_val`, por defecto 1.0). ∞ si son idénticas."""
    a = _as_batch(x).float()
    b = _as_batch(xhat).float()
    per_img_mse = torch.mean((a - b) ** 2, dim=[1, 2, 3])
    per_img_psnr = 10.0 * torch.log10((max_val ** 2) / (per_img_mse + EPS))
    return float(per_img_psnr.mean().item())


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 3) -> torch.Tensor:
    """Kernel gaussiano separable 2D normalizado, replicado por canal: (C,1,k,k)."""
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    kernel_2d = (g.t() @ g)  # (k,k)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    x: torch.Tensor,
    xhat: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> float:
    """SSIM medio con ventana gaussiana (implementado a mano). Imágenes en [0,1].

    Sigue la formulación estándar de Wang et al. (2004): medias, varianzas y covarianza locales
    estimadas con una ventana gaussiana, y las constantes de estabilidad C1=(0.01·L)², C2=(0.03·L)².
    """
    a = _as_batch(x).float()
    b = _as_batch(xhat).float()
    channels = a.shape[1]
    window = _gaussian_window(window_size, sigma, channels).to(a.device, a.dtype)
    pad = window_size // 2

    def _filter(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window, padding=pad, groups=channels)

    mu_a = _filter(a)
    mu_b = _filter(b)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sigma_a2 = _filter(a * a) - mu_a2
    sigma_b2 = _filter(b * b) - mu_b2
    sigma_ab = _filter(a * b) - mu_ab

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    ssim_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    # media sobre canales y espacio, luego sobre el lote
    return float(ssim_map.mean(dim=[1, 2, 3]).mean().item())
