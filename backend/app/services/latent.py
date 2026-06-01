"""Proyección 2D (PCA) y clustering (k-means) sobre vectores latentes z.

Reutilizable por AE y VAE. PCA por defecto (CLAUDE.md §3.7); UMAP queda opcional/diferido:
si se pide y no está instalado, se degrada a PCA y se avisa con `method_used`.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def project_2d(Z: np.ndarray, method: str = "pca", seed: int = 42) -> tuple[np.ndarray, str]:
    """Devuelve (coords (N,2), método_usado). Cae a PCA si se pide UMAP y no está disponible."""
    if method == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(n_components=2, random_state=seed)
            return reducer.fit_transform(Z), "umap"
        except Exception:
            pass  # degradar a PCA
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(Z), "pca"


def kmeans_labels(Z: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=int(n_clusters), random_state=seed, n_init=10)
    return km.fit_predict(Z)
