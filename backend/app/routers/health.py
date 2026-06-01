"""Endpoint común /api/health — informa del device detectado y versiones."""

import os
import sys

import torch
from fastapi import APIRouter
from pydantic import BaseModel

from ..device import device_label, get_device

router = APIRouter(tags=["common"])


class Health(BaseModel):
    status: str
    device: str  # "mps" | "cpu"
    device_name: str
    torch_version: str
    python_version: str
    mps_available: bool
    mps_fallback: bool


@router.get("/health", response_model=Health)
def health() -> Health:
    device = get_device()
    return Health(
        status="ok",
        device=device.type,
        device_name=device_label(device),
        torch_version=torch.__version__,
        python_version=sys.version.split()[0],
        mps_available=torch.backends.mps.is_available(),
        mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
    )
