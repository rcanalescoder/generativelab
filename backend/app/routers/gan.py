"""Endpoints del GAN (CLAUDE.md §4.3, §5). Entrenamiento por SSE; el resto REST.

Un GAN no reconstruye ni tiene vecinos/mapa latente clásico: expone generación desde
ruido z e interpolación en ese mismo espacio de ruido.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from queue import Empty

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..services.gan_service import GANHyperParams, gan_service

router = APIRouter(prefix="/gan", tags=["gan"])


class HyperParamsIn(BaseModel):
    z_dim: int | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    batch_size: int | None = None
    arch: str | None = None


class TrainRequest(BaseModel):
    mode: str = "quick"  # "full" | "quick"
    seed: int = 42
    hyperparams: HyperParamsIn = Field(default_factory=HyperParamsIn)


class GenerateRequest(BaseModel):
    n: int = 16
    seed: int = 42


class InterpolateRequest(BaseModel):
    steps: int = 8
    seed: int = 42


class ArchRequest(BaseModel):
    arch: str  # "basico" | "grande"


def _require_model() -> None:
    if gan_service.generator is None:
        raise HTTPException(status_code=409, detail="modelo no entrenado: entrena o carga el checkpoint demo")


@router.get("/status")
def status() -> dict:
    return gan_service.status()


@router.post("/hyperparams")
def set_hyperparams(req: HyperParamsIn) -> dict:
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    return gan_service.set_hyperparams(patch)


@router.post("/arch")
def set_arch(req: ArchRequest) -> dict:
    """Cambia de variante cargando su checkpoint demo al instante (si existe).

    Devuelve `{loaded, ...status}`: `loaded=False` si esa variante aún no tiene demo entrenado
    (el front puede entonces ofrecer entrenarla). Si carga, el status refleja la nueva arch.
    """
    loaded = gan_service.load_demo(req.arch)
    if not loaded:
        # No hay demo de esa variante: deja registrada la elección como hiperparámetro.
        gan_service.set_hyperparams({"arch": req.arch})
    return {"loaded": loaded, **gan_service.status()}


@router.post("/train")
async def train(req: TrainRequest, request: Request) -> StreamingResponse:
    merged = {**asdict(gan_service.hp), **{k: v for k, v in req.hyperparams.model_dump().items() if v is not None}}
    hp = GANHyperParams(**merged)
    events = gan_service.start_training(req.mode, hp, req.seed)

    async def gen():
        while True:
            try:
                ev = await run_in_threadpool(events.get, True, 1.0)
            except Empty:
                # si el cliente se va, cancelamos el entrenamiento y cerramos
                if await request.is_disconnected():
                    gan_service.cancel()
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
    gan_service.cancel()
    return {"cancelled": True}


@router.post("/generate")
def generate(req: GenerateRequest) -> dict:
    _require_model()
    return {"images": gan_service.generate(req.n, req.seed)}


@router.post("/interpolate")
def interpolate(req: InterpolateRequest) -> dict:
    _require_model()
    return gan_service.interpolate(req.steps, req.seed)
