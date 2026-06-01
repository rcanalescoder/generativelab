"""FastAPI app: CORS + routers. Fase 0 expone solo /api/health."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .device import device_label, get_device
from .routers import autoencoder, dataset, diffusion, gan, health, upload, vae
from .services.ae_service import ae_service
from .services.diffusion_service import diffusion_service
from .services.gan_service import gan_service
from .services.vae_service import vae_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    device = get_device()
    print(f"[GMVL] Device: {device.type} ({device_label(device)}) · PyTorch {torch.__version__}")
    # Modo demo: cargar el checkpoint preentrenado si existe (CLAUDE.md §3.2).
    if ae_service.load_checkpoint():
        print(f"[GMVL] Autoencoder: checkpoint demo cargado (latent_dim={ae_service.hp.latent_dim})")
    else:
        print("[GMVL] Autoencoder: sin checkpoint demo (entrena para generarlo)")
    for label, svc in (("VAE", vae_service), ("GAN", gan_service), ("Diffusion", diffusion_service)):
        try:
            loaded = svc.load_checkpoint()
        except Exception as exc:  # pragma: no cover - defensivo
            print(f"[GMVL] {label}: error cargando checkpoint ({exc})")
            loaded = False
        print(f"[GMVL] {label}: {'checkpoint demo cargado' if loaded else 'sin checkpoint demo (entrena para generarlo)'}")
    yield


app = FastAPI(
    title="Generative Models Visual Lab API",
    version="0.1.0",
    lifespan=lifespan,
)

# El front de desarrollo vive en Vite (:5173). También se permite el proxy same-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(dataset.router, prefix="/api")
app.include_router(autoencoder.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(vae.router, prefix="/api")
app.include_router(gan.router, prefix="/api")
app.include_router(diffusion.router, prefix="/api")
