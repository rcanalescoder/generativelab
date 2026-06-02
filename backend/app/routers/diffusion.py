"""Endpoints de Diffusion (CLAUDE.md §5). Entrenamiento por SSE; el resto REST.

NOTA DE INTEGRACIÓN: este router se registra en `app/main.py` con
    from .routers import diffusion
    app.include_router(diffusion.router, prefix="/api")
y, opcionalmente, en el `lifespan` se puede llamar a `diffusion_service.load_checkpoint()`
para el modo demo. Como respaldo, `_require_model` intenta cargar el checkpoint demo de
forma perezosa la primera vez que se necesita el modelo.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from queue import Empty

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..services.diffusion_service import DiffusionHyperParams, diffusion_service

router = APIRouter(prefix="/diffusion", tags=["diffusion"])


class HyperParamsIn(BaseModel):
    learning_rate: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    timesteps: int | None = None
    schedule: str | None = None  # "cosine" (def.) | "linear"
    arch: str | None = None      # "agil" | "nitido"


class ArchRequest(BaseModel):
    arch: str  # "agil" | "nitido"


class TrainRequest(BaseModel):
    mode: str = "quick"  # "full" | "quick"
    seed: int = 42
    hyperparams: HyperParamsIn = Field(default_factory=HyperParamsIn)
    early_stop: bool = True


class GenerateRequest(BaseModel):
    n: int = 6
    seed: int = 42
    steps: int = 80  # más pasos = mejor calidad de muestra (submuestreo de T)


class TrajectoryRequest(BaseModel):
    seed: int = 42
    steps: int = 80  # más pasos = mejor calidad de muestra (submuestreo de T)
    snapshots: int = 8


def _require_model() -> None:
    if diffusion_service.model is None:
        # respaldo: intentar cargar el checkpoint demo de forma perezosa
        if not diffusion_service.load_checkpoint():
            raise HTTPException(
                status_code=409,
                detail="modelo no entrenado: entrena o genera el checkpoint demo",
            )


@router.get("/status")
def status() -> dict:
    return diffusion_service.status()


@router.post("/arch")
def set_arch(req: ArchRequest) -> dict:
    """Cambia de variante cargando su checkpoint demo al instante (si existe).

    Devuelve `{loaded, ...status}`: `loaded=False` si esa variante aún no tiene demo entrenado
    (el front puede entonces ofrecer entrenarla). Si carga, el status refleja la nueva arch.
    """
    loaded = diffusion_service.load_demo(req.arch)
    if not loaded:
        # No hay demo de esa variante: deja registrada la elección como hiperparámetro
        # (saneada contra DIFF_ARCHS) para que /train la propague.
        diffusion_service.hp = replace(diffusion_service.hp, arch=req.arch).sanitized()
    return {"loaded": loaded, **diffusion_service.status()}


@router.post("/train")
async def train(req: TrainRequest, request: Request) -> StreamingResponse:
    merged = {
        **asdict(diffusion_service.hp),
        **{k: v for k, v in req.hyperparams.model_dump().items() if v is not None},
    }
    hp = DiffusionHyperParams(**merged)
    events = diffusion_service.start_training(req.mode, hp, req.seed, req.early_stop)

    async def gen():
        while True:
            try:
                ev = await run_in_threadpool(events.get, True, 1.0)
            except Empty:
                # si el cliente se va, cancelamos el entrenamiento y cerramos
                if await request.is_disconnected():
                    diffusion_service.cancel()
                    break
                continue
            if ev is None:  # centinela de fin
                break
            yield f"data: {json.dumps(ev)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/train/cancel")
def cancel() -> dict:
    diffusion_service.cancel()
    return {"cancelled": True}


@router.post("/generate")
def generate(req: GenerateRequest) -> dict:
    _require_model()
    return {"images": diffusion_service.generate(req.n, req.seed, req.steps)}


@router.post("/trajectory")
def trajectory(req: TrajectoryRequest) -> dict:
    _require_model()
    return {"frames": diffusion_service.trajectory(req.seed, req.steps, req.snapshots)}
