"""Carga del dataset huggan/anime-faces con cache local y fallback (CLAUDE.md §3.6).

Estrategia:
1. Descargar `data.zip` UNA vez (lo cachea huggingface_hub en ~/.cache/huggingface).
2. Decodificar las 21.551 PNG a un array uint8 (N, 64, 64, 3) y guardarlo en
   `backend/data/anime-faces/faces_64.npy`. A partir de ahí, todo es local y rápido.
3. Si no hay red pero existe una carpeta local de imágenes, se construye desde ahí (fallback).

No se usa el *streaming* del viewer de HuggingFace (inestable). El id de cada imagen es su
índice en el array (orden estable: por el número del nombre de fichero).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ID = "huggan/anime-faces"
ZIP_FILENAME = "data.zip"
IMG_SIZE = 64

# backend/data/anime-faces/  (data.py está en backend/app/data/)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "anime-faces"
ARRAY_PATH = DATA_DIR / "faces_64.npy"
LOCAL_IMAGES_DIR = DATA_DIR / "images"

_NUM_RE = re.compile(r"(\d+)")
_cache: np.ndarray | None = None


def _numeric_key(name: str) -> tuple[int, str]:
    """Ordena '10.png' después de '2.png' (por el número, no lexicográficamente)."""
    m = _NUM_RE.search(Path(name).stem)
    return (int(m.group(1)) if m else 0, name)


def _decode(raw: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _build_from_zip(zip_path: Path, verbose: bool) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(
            (n for n in zf.namelist() if n.lower().endswith(".png")),
            key=_numeric_key,
        )
        n = len(names)
        if verbose:
            print(f"[dataset] {n} PNG en {zip_path.name}; decodificando…", flush=True)
        arr = np.empty((n, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for i, name in enumerate(names):
            arr[i] = _decode(zf.read(name))
            if verbose and (i + 1) % 2000 == 0:
                print(f"[dataset]   {i + 1}/{n}", flush=True)
    return arr


def _build_from_local_dir(dir_path: Path, verbose: bool) -> np.ndarray:
    paths = sorted(
        (p for p in dir_path.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: _numeric_key(p.name),
    )
    if not paths:
        raise FileNotFoundError(f"No hay imágenes en {dir_path}")
    if verbose:
        print(f"[dataset] {len(paths)} imágenes locales en {dir_path}", flush=True)
    arr = np.empty((len(paths), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        arr[i] = _decode(p.read_bytes())
    return arr


def ensure_array(verbose: bool = False) -> Path:
    """Garantiza que existe faces_64.npy. Descarga + construye si hace falta. Devuelve la ruta."""
    if ARRAY_PATH.exists():
        if verbose:
            print(f"[dataset] cache ya presente: {ARRAY_PATH}", flush=True)
        return ARRAY_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Fallback offline: carpeta local de imágenes.
    if LOCAL_IMAGES_DIR.is_dir():
        try:
            arr = _build_from_local_dir(LOCAL_IMAGES_DIR, verbose)
            np.save(ARRAY_PATH, arr)
            if verbose:
                print(f"[dataset] guardado {ARRAY_PATH} {arr.shape} (desde carpeta local)", flush=True)
            return ARRAY_PATH
        except FileNotFoundError:
            pass

    # Camino normal: descargar el zip (cacheado) y construir.
    from huggingface_hub import hf_hub_download

    if verbose:
        print(f"[dataset] descargando {REPO_ID}/{ZIP_FILENAME} (~462 MB, solo la 1ª vez)…", flush=True)
    zip_path = Path(
        hf_hub_download(REPO_ID, ZIP_FILENAME, repo_type="dataset")
    )
    arr = _build_from_zip(zip_path, verbose)
    np.save(ARRAY_PATH, arr)
    if verbose:
        print(f"[dataset] guardado {ARRAY_PATH} {arr.shape}", flush=True)
    return ARRAY_PATH


def load_array(mmap: bool = True) -> np.ndarray:
    """Carga (y cachea en memoria) el array uint8 (N,64,64,3). mmap para no duplicar RAM."""
    global _cache
    if _cache is None:
        ensure_array(verbose=False)
        _cache = np.load(ARRAY_PATH, mmap_mode="r" if mmap else None)
    return _cache


def is_ready() -> bool:
    return ARRAY_PATH.exists()


def count() -> int:
    return int(load_array().shape[0])


def get_image_uint8(idx: int) -> np.ndarray:
    """Imagen (64,64,3) uint8 por id (= índice)."""
    arr = load_array()
    return np.asarray(arr[idx])


if __name__ == "__main__":
    # Permite preparar el dataset por adelantado:  python -m app.data.dataset
    ensure_array(verbose=True)
    print("[dataset] listo:", count(), "imágenes")
