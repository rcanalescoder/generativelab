"""Experimento del Autoencoder: comparar variantes y latent_dim (bucle de realimentación).

Entrena, sobre un subconjunto y por pocas epochs, las configuraciones:
  - las tres variantes (basico/grande/unet) a latent_dim 128,
  - "basico" además a latent_dim 64 y 256 (efecto del cuello de botella).

Evalúa cada una con MSE/PSNR/SSIM sobre el MISMO held-out fijo que usa la app
(`AEService.metrics`), imprime una tabla y escribe `ae_search_results.json` al lado de este
fichero. NO toca los checkpoints demo ni el estado del servidor.

Uso:
    cd backend && .venv/bin/python -m app.experiments.ae_search
    # opcional: EPOCHS=8 TRAIN_N=8000 python -m app.experiments.ae_search
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..models.ae import count_params
from ..services.ae_service import AEHyperParams, AEService

# Configuraciones a evaluar: (arch, latent_dim).
CONFIGS: list[tuple[str, int]] = [
    ("basico", 64),
    ("basico", 128),
    ("basico", 256),
    ("grande", 128),
    ("unet", 128),
]

EPOCHS = int(os.environ.get("EPOCHS", 8))      # epochs por configuración (acotado)
TRAIN_N = int(os.environ.get("TRAIN_N", 8000))  # tamaño del subconjunto de entrenamiento
SEED = int(os.environ.get("SEED", 42))
EVAL_N = int(os.environ.get("EVAL_N", 256))     # held-out fijo (idéntico al de la app)

RESULTS_PATH = Path(__file__).resolve().parent / "ae_search_results.json"


def _train_one(arch: str, latent_dim: int) -> dict:
    """Entrena una configuración en modo 'quick' (subconjunto) y la evalúa en el held-out fijo."""
    svc = AEService()
    # QUICK_N controla el tamaño del subconjunto en modo 'quick'; lo ajustamos a TRAIN_N.
    import app.services.ae_service as ae_mod

    ae_mod.QUICK_N = TRAIN_N
    hp = AEHyperParams(latent_dim=latent_dim, epochs=EPOCHS, batch_size=256, arch=arch)

    t0 = time.time()
    final_loss = None
    epochs_run = 0
    for ev in svc.iter_train("quick", hp, seed=SEED, early_stop=False):
        if ev["type"] == "epoch":
            epochs_run = ev["epoch"]
            final_loss = ev["loss"]
            print(f"    [{arch} ld={latent_dim}] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}",
                  flush=True)
    train_s = time.time() - t0

    m = svc.metrics(n=EVAL_N)
    return {
        "arch": arch,
        "latent_dim": latent_dim,
        "params": int(count_params(svc.model)) if svc.model is not None else 0,
        "train_epochs": epochs_run,
        "train_loss": final_loss,
        "train_seconds": round(train_s, 1),
        "eval_n": m["n"],
        "mse": m["mse"],
        "psnr": m["psnr"],
        "ssim": m["ssim"],
    }


def main() -> None:
    print(f"[ae-search] {len(CONFIGS)} configs · epochs={EPOCHS} · train_n={TRAIN_N} · "
          f"eval_n={EVAL_N} · seed={SEED}", flush=True)
    results: list[dict] = []
    for arch, ld in CONFIGS:
        print(f"[ae-search] === {arch} · latent_dim {ld} ===", flush=True)
        results.append(_train_one(arch, ld))

    # Ordena por PSNR descendente (mejor reconstrucción primero).
    results.sort(key=lambda r: r["psnr"], reverse=True)

    print("\n[ae-search] === Tabla de resultados (held-out fijo, ordenado por PSNR) ===", flush=True)
    header = f"{'arch':8s} {'ld':>4s} {'params':>10s} {'MSE':>10s} {'PSNR(dB)':>9s} {'SSIM':>7s} {'train(s)':>9s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in results:
        print(f"{r['arch']:8s} {r['latent_dim']:>4d} {r['params'] / 1e6:9.2f}M "
              f"{r['mse']:10.5f} {r['psnr']:9.2f} {r['ssim']:7.4f} {r['train_seconds']:9.1f}",
              flush=True)

    payload = {
        "config": {"epochs": EPOCHS, "train_n": TRAIN_N, "eval_n": EVAL_N, "seed": SEED},
        "results": results,
        "best_psnr": results[0] if results else None,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[ae-search] resultados escritos en {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
