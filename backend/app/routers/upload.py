"""POST /api/upload — reconstruye una imagen subida con el AE (CLAUDE.md §5)."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from .. import imaging
from ..services.ae_service import ae_service

router = APIRouter(tags=["common"])


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        x = imaging.load_upload_tensor(raw)
    except Exception:
        raise HTTPException(status_code=400, detail="imagen no válida")
    if ae_service.model is None:
        raise HTTPException(status_code=409, detail="modelo no entrenado")
    return ae_service.reconstruct_tensor(x)
