"""Experimento del GAN: comparar las variantes de arquitectura (bucle de realimentación).

Entrena, sobre un subconjunto y por pocas epochs, las dos variantes del GAN
(basico/grande) y, como el GAN **no tiene métrica de reconstrucción**, la comparación es
**visual**: genera una pequeña galería de caras desde un z fijo (la misma semilla para todas
las variantes, para que sean comparables) y la escribe como PNGs + un índice JSON al lado de
este fichero. NO toca los checkpoints demo ni el estado del servidor.

Uso:
    cd backend && .venv/bin/python -m app.experiments.gan_search
    # opcional: EPOCHS=15 TRAIN_N=8000 N_SAMPLES=12 python -m app.experiments.gan_search
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import torch

from ..models.gan import build_gan, count_params, to_image_range
from ..services.gan_service import GANHyperParams, GANService

# Variantes a evaluar (todas las del modelo).
from ..models.gan import GAN_ARCHS as CONFIGS

EPOCHS = int(os.environ.get("EPOCHS", 15))       # epochs por variante (acotado)
TRAIN_N = int(os.environ.get("TRAIN_N", 8000))   # tamaño del subconjunto de entrenamiento
SEED = int(os.environ.get("SEED", 42))           # semilla de entrenamiento
SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", 7))  # semilla de la galería (igual para todas)
N_SAMPLES = int(os.environ.get("N_SAMPLES", 12))  # nº de caras generadas por variante
Z_DIM = int(os.environ.get("Z_DIM", 100))

OUT_DIR = Path(__file__).resolve().parent / "gan_search_out"
RESULTS_PATH = OUT_DIR / "gan_search_results.json"


def _train_one(arch: str) -> dict:
    """Entrena una variante en modo 'quick' (subconjunto) y genera su galería visual."""
    svc = GANService()
    # QUICK_N controla el tamaño del subconjunto en modo 'quick'; lo ajustamos a TRAIN_N.
    import app.services.gan_service as gan_mod

    gan_mod.QUICK_N = TRAIN_N
    hp = GANHyperParams(z_dim=Z_DIM, epochs=EPOCHS, batch_size=128, arch=arch)

    t0 = time.time()
    g_loss = d_loss = None
    epochs_run = 0
    for ev in svc.iter_train("quick", hp, seed=SEED):
        if ev["type"] == "epoch":
            epochs_run = ev["epoch"]
            g_loss, d_loss = ev["g_loss"], ev["d_loss"]
            print(f"    [{arch}] epoch {ev['epoch']}/{ev['epochs']} · "
                  f"g_loss {ev['g_loss']:.4f} · d_loss {ev['d_loss']:.4f}", flush=True)
    train_s = time.time() - t0

    # Galería visual: z fijo (misma semilla para todas las variantes → comparables).
    assert svc.generator is not None
    g = torch.Generator(device="cpu").manual_seed(SAMPLE_SEED)
    z = torch.randn(N_SAMPLES, Z_DIM, generator=g).to(svc.device)
    svc.generator.eval()
    with torch.no_grad():
        imgs = to_image_range(svc.generator(z)).cpu()

    files: list[str] = []
    for k in range(N_SAMPLES):
        from .. import imaging

        b64 = imaging.tensor_to_b64(imgs[k])  # data URL "data:image/png;base64,..."
        png = base64.b64decode(b64.split(",", 1)[1])
        fname = f"{arch}_{k:02d}.png"
        (OUT_DIR / fname).write_bytes(png)
        files.append(fname)

    return {
        "arch": arch,
        "params": int(count_params(svc.generator) + count_params(svc.discriminator))
        if svc.discriminator is not None else int(count_params(svc.generator)),
        "params_g": int(count_params(svc.generator)),
        "train_epochs": epochs_run,
        "g_loss": g_loss,
        "d_loss": d_loss,
        "train_seconds": round(train_s, 1),
        "sample_files": files,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[gan-search] {len(CONFIGS)} variantes · epochs={EPOCHS} · train_n={TRAIN_N} · "
          f"seed={SEED} · galería n={N_SAMPLES} (sample_seed={SAMPLE_SEED})", flush=True)
    results: list[dict] = []
    for arch in CONFIGS:
        print(f"[gan-search] === variante '{arch}' ===", flush=True)
        results.append(_train_one(arch))

    print("\n[gan-search] === Resumen (la comparación de calidad es VISUAL: mira los PNG) ===",
          flush=True)
    header = f"{'arch':8s} {'params':>10s} {'g_loss':>9s} {'d_loss':>9s} {'train(s)':>9s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        gl = f"{r['g_loss']:.4f}" if r["g_loss"] is not None else "—"
        dl = f"{r['d_loss']:.4f}" if r["d_loss"] is not None else "—"
        print(f"{r['arch']:8s} {r['params'] / 1e6:9.2f}M {gl:>9s} {dl:>9s} "
              f"{r['train_seconds']:9.1f}", flush=True)

    payload = {
        "config": {"epochs": EPOCHS, "train_n": TRAIN_N, "seed": SEED,
                   "sample_seed": SAMPLE_SEED, "n_samples": N_SAMPLES, "z_dim": Z_DIM},
        "note": "El GAN no tiene métrica de reconstrucción; la comparación es visual (galería PNG).",
        "results": results,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[gan-search] galerías y resultados escritos en {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
