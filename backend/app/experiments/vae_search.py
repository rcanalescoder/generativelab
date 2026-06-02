"""Experimento del VAE: comparar variantes y latent_dim (bucle de realimentación).

Entrena, sobre un subconjunto y por pocas epochs (con un β fijo), las configuraciones:
  - las dos variantes (basico/grande) a latent_dim 128,
  - "basico" además a latent_dim 64 y 256 (efecto del cuello de botella).

Evalúa cada una con MSE/PSNR/SSIM de reconstrucción (vía μ) sobre el MISMO held-out fijo que usa
la app (`VAEService.metrics`), imprime una tabla y escribe `vae_search_results.json` al lado de
este fichero. NO toca los checkpoints demo ni el estado del servidor.

Uso:
    cd backend && .venv/bin/python -m app.experiments.vae_search
    # opcional: EPOCHS=8 TRAIN_N=8000 BETA=1.0 python -m app.experiments.vae_search
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..models.vae import count_params
from ..services.vae_service import VAEHyperParams, VAEService

# Configuraciones a evaluar: (arch, latent_dim).
CONFIGS: list[tuple[str, int]] = [
    ("basico", 64),
    ("basico", 128),
    ("basico", 256),
    ("grande", 128),
]

EPOCHS = int(os.environ.get("EPOCHS", 8))       # epochs por configuración (acotado)
TRAIN_N = int(os.environ.get("TRAIN_N", 8000))  # tamaño del subconjunto de entrenamiento
SEED = int(os.environ.get("SEED", 42))
EVAL_N = int(os.environ.get("EVAL_N", 256))     # held-out fijo (idéntico al de la app)
BETA = float(os.environ.get("BETA", 1.0))       # peso del término KL (β-VAE) para todas las configs

RESULTS_PATH = Path(__file__).resolve().parent / "vae_search_results.json"


def _train_one(arch: str, latent_dim: int) -> dict:
    """Entrena una configuración en modo 'quick' (subconjunto) y la evalúa en el held-out fijo."""
    svc = VAEService()
    # QUICK_N controla el tamaño del subconjunto en modo 'quick'; lo ajustamos a TRAIN_N.
    import app.services.vae_service as vae_mod

    vae_mod.QUICK_N = TRAIN_N
    hp = VAEHyperParams(latent_dim=latent_dim, epochs=EPOCHS, beta=BETA, batch_size=256, arch=arch)

    t0 = time.time()
    final_loss = None
    final_recon = None
    final_kl = None
    epochs_run = 0
    for ev in svc.iter_train("quick", hp, seed=SEED, early_stop=False):
        if ev["type"] == "epoch":
            epochs_run = ev["epoch"]
            final_loss = ev["loss"]
            final_recon = ev["recon_loss"]
            final_kl = ev["kl_loss"]
            print(
                f"    [{arch} ld={latent_dim}] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}"
                f" (recon {ev['recon_loss']:.5f} · kl {ev['kl_loss']:.5f})",
                flush=True,
            )
    train_s = time.time() - t0

    m = svc.metrics(n=EVAL_N)
    return {
        "arch": arch,
        "latent_dim": latent_dim,
        "beta": BETA,
        "params": int(count_params(svc.model)) if svc.model is not None else 0,
        "train_epochs": epochs_run,
        "train_loss": final_loss,
        "train_recon": final_recon,
        "train_kl": final_kl,
        "train_seconds": round(train_s, 1),
        "eval_n": m["n"],
        "mse": m["mse"],
        "psnr": m["psnr"],
        "ssim": m["ssim"],
    }


def main() -> None:
    print(f"[vae-search] {len(CONFIGS)} configs · epochs={EPOCHS} · train_n={TRAIN_N} · "
          f"eval_n={EVAL_N} · beta={BETA} · seed={SEED}", flush=True)
    results: list[dict] = []
    for arch, ld in CONFIGS:
        print(f"[vae-search] === {arch} · latent_dim {ld} ===", flush=True)
        results.append(_train_one(arch, ld))

    # Ordena por PSNR descendente (mejor reconstrucción primero).
    results.sort(key=lambda r: r["psnr"], reverse=True)

    print("\n[vae-search] === Tabla de resultados (held-out fijo, ordenado por PSNR) ===", flush=True)
    header = f"{'arch':8s} {'ld':>4s} {'params':>10s} {'MSE':>10s} {'PSNR(dB)':>9s} {'SSIM':>7s} {'train(s)':>9s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        print(f"{r['arch']:8s} {r['latent_dim']:>4d} {r['params'] / 1e6:9.2f}M "
              f"{r['mse']:10.5f} {r['psnr']:9.2f} {r['ssim']:7.4f} {r['train_seconds']:9.1f}",
              flush=True)

    payload = {
        "config": {"epochs": EPOCHS, "train_n": TRAIN_N, "eval_n": EVAL_N, "beta": BETA, "seed": SEED},
        "results": results,
        "best_psnr": results[0] if results else None,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[vae-search] resultados escritos en {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
