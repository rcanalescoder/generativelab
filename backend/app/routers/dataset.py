"""Endpoints comunes del dataset (CLAUDE.md §5)."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .. import imaging
from ..data import dataset

router = APIRouter(prefix="/dataset", tags=["dataset"])


class DatasetInfo(BaseModel):
    name: str
    count: int
    image_shape: list[int]
    license: str


@router.get("/info", response_model=DatasetInfo)
def info() -> DatasetInfo:
    return DatasetInfo(
        name="huggan/anime-faces",
        count=dataset.count(),
        image_shape=[64, 64, 3],
        license="CC0 1.0",
    )


@router.get("/samples")
def samples(n: int = Query(12, ge=1, le=64), seed: int = 0) -> dict:
    arr = dataset.load_array()
    total = dataset.count()
    rng = np.random.default_rng(seed)
    ids = sorted(int(i) for i in rng.choice(total, size=min(n, total), replace=False))
    return {"samples": [{"id": i, "image": imaging.encode_png_b64(np.asarray(arr[i]))} for i in ids]}


@router.get("/image/{idx}")
def image(idx: int) -> dict:
    total = dataset.count()
    if idx < 0 or idx >= total:
        raise HTTPException(status_code=404, detail="id fuera de rango")
    return {"id": idx, "image": imaging.encode_png_b64(np.asarray(dataset.load_array()[idx]))}
