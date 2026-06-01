"""Conversión imagen ↔ tensor ↔ PNG base64 (data URL). Compartido por los endpoints.

Convención: imágenes como PNG base64 (data URL) listas para usar en <img src> (CLAUDE.md §5).
Tensores de imagen: float CHW en [0,1].
"""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from PIL import Image

IMG_SIZE = 64


def encode_png_b64(arr_uint8_hwc: np.ndarray) -> str:
    """(H,W,3) uint8 → data URL PNG."""
    img = Image.fromarray(arr_uint8_hwc, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def tensor_to_b64(t: torch.Tensor) -> str:
    """Tensor CHW float [0,1] → data URL PNG."""
    arr = t.detach().clamp(0, 1).mul(255).round().to(torch.uint8)
    arr = arr.permute(1, 2, 0).cpu().numpy()
    return encode_png_b64(arr)


def uint8_to_tensor(arr_uint8_hwc: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → tensor CHW float [0,1]."""
    # copy=True: el array puede venir de un mmap de solo lectura.
    arr = np.array(arr_uint8_hwc, dtype=np.uint8, copy=True)
    t = torch.from_numpy(arr).float().div(255)
    return t.permute(2, 0, 1)


def diff_b64(x: torch.Tensor, x_hat: torch.Tensor, gain: float = 3.0) -> str:
    """Mapa |x − x̂| sobre fondo oscuro, tintado en rosa donde hay error (estética de la maqueta)."""
    d = (x - x_hat).abs().mean(0)  # (H,W)
    d = (d * gain).clamp(0, 1).cpu().numpy()
    navy = np.array([25, 23, 48], dtype=np.float32)
    pink = np.array([236, 72, 153], dtype=np.float32)
    rgb = navy[None, None, :] * (1 - d[..., None]) + pink[None, None, :] * d[..., None]
    return encode_png_b64(rgb.astype(np.uint8))


def load_upload_tensor(raw: bytes) -> torch.Tensor:
    """Bytes de una imagen subida → tensor CHW [0,1] redimensionado a 64×64 RGB."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return uint8_to_tensor(np.asarray(img, dtype=np.uint8))
