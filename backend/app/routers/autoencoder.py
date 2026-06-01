"""Endpoints del Autoencoder (CLAUDE.md §5). Entrenamiento por SSE; el resto REST."""

from __future__ import annotations

import json
from dataclasses import asdict
from queue import Empty

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from ..services.ae_service import AEHyperParams, ae_service

router = APIRouter(prefix="/autoencoder", tags=["autoencoder"])


class HyperParamsIn(BaseModel):
    latent_dim: int | None = None
    learning_rate: float | None = None
    epochs: int | None = None
    loss: str | None = None
    batch_size: int | None = None


class TrainRequest(BaseModel):
    mode: str = "quick"  # "full" | "quick"
    seed: int = 42
    hyperparams: HyperParamsIn = Field(default_factory=HyperParamsIn)


class ReconstructRequest(BaseModel):
    ids: list[int]
    noise: float = 0.0


class NeighborsRequest(BaseModel):
    query_id: int
    k: int = 8


class ClusterRequest(BaseModel):
    n_clusters: int = 5


class InterpolateRequest(BaseModel):
    a_id: int
    b_id: int
    steps: int = 6


def _require_model() -> None:
    if ae_service.model is None:
        raise HTTPException(status_code=409, detail="modelo no entrenado: entrena o carga el checkpoint demo")


@router.get("/status")
def status() -> dict:
    return ae_service.status()


@router.post("/hyperparams")
def set_hyperparams(req: HyperParamsIn) -> dict:
    patch = {k: v for k, v in req.model_dump().items() if v is not None}
    return ae_service.set_hyperparams(patch)


@router.post("/train")
async def train(req: TrainRequest, request: Request) -> StreamingResponse:
    merged = {**asdict(ae_service.hp), **{k: v for k, v in req.hyperparams.model_dump().items() if v is not None}}
    hp = AEHyperParams(**merged)
    events = ae_service.start_training(req.mode, hp, req.seed)

    async def gen():
        while True:
            try:
                ev = await run_in_threadpool(events.get, True, 1.0)
            except Empty:
                # si el cliente se va, cancelamos el entrenamiento y cerramos
                if await request.is_disconnected():
                    ae_service.cancel()
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
    ae_service.cancel()
    return {"cancelled": True}


@router.post("/reconstruct")
def reconstruct(req: ReconstructRequest) -> dict:
    _require_model()
    return {"items": ae_service.reconstruct(req.ids, req.noise)}


@router.get("/projection")
def projection(method: str = Query("pca", pattern="^(pca|umap)$")) -> dict:
    _require_model()
    return ae_service.projection(method)


@router.post("/neighbors")
def neighbors(req: NeighborsRequest) -> dict:
    _require_model()
    return ae_service.neighbors(req.query_id, req.k)


@router.post("/cluster")
def cluster(req: ClusterRequest) -> dict:
    _require_model()
    return ae_service.cluster(req.n_clusters)


@router.post("/interpolate")
def interpolate(req: InterpolateRequest) -> dict:
    _require_model()
    return ae_service.interpolate(req.a_id, req.b_id, req.steps)
