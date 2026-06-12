"""Entrenamiento LARGO y reanudable de diffusion «nitido» (64×64) — ciclo v3.

Mejoras frente a `_train_nitido.py` (v2): dataset completo por defecto (43k vs 18k),
warmup de lr, guardado periódico DOBLE (demo para la app + estado completo para reanudar)
y rejilla de progreso por guardado para inspección visual.

  cd backend && PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m app.experiments.train_diffusion_v3 \
      --epochs 24 [--batch 64] [--n 0] [--resume] [--save-every 4]

  --n 0       → dataset completo (por defecto); >0 → subconjunto
  --resume    → continúa desde experiments/ckpts/diffusion_nitido_v3_state.pt
  cada guardado actualiza checkpoints/diffusion_nitido_demo.pt (pesos EMA → lo carga la app
  y lo evalúa `runner.py --load-demo nitido`).
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data import dataset
from ..device import get_device
from ..models.diffusion import EMA, DiffusionSchedule, build_diffusion, q_sample, sample_loop
from ..seeding import set_seed
from . import quality

STATE_PATH = quality.EXP_DIR / "ckpts" / "diffusion_nitido_v3_state.pt"
WARMUP_STEPS = 500   # warmup lineal de lr (estabiliza los primeros pasos de la UNet)
T = 200              # mismo T del servicio (compatibilidad con la app)
SEED = 42


def _save_demo(model_ema: torch.nn.Module, epochs_done: int, loss: float, lr: float) -> None:
    """Escribe el checkpoint demo (pesos EMA) en el formato que carga la app."""
    import app.services.diffusion_service as D

    svc = D.DiffusionService()
    svc.model = model_ema
    svc.hp = D.DiffusionHyperParams(learning_rate=lr, epochs=epochs_done, schedule="cosine",
                                    timesteps=T, arch="nitido").sanitized()
    svc.seed = SEED
    svc.trained = True
    svc.loss_history = [{"epoch": epochs_done, "loss": loss}]
    svc.save_checkpoint(D.demo_path("nitido"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=0, help="0 = dataset completo")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    device = get_device()
    set_seed(SEED)
    model = build_diffusion("nitido").to(device)
    model.train()
    sched = DiffusionSchedule(T, device, "cosine")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=0.999)
    ema_model = build_diffusion("nitido").to(device)
    ema_model.eval()

    start_epoch = 1
    gstep = 0
    if args.resume and STATE_PATH.exists():
        st = torch.load(STATE_PATH, map_location=device)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        ema.shadow = {k: v.to(device) for k, v in st["ema"].items()}
        start_epoch = int(st["epoch"]) + 1
        gstep = int(st.get("gstep", 0))
        print(f"[nitido-v3] REANUDANDO en epoch {start_epoch} (gstep {gstep})", flush=True)

    ntot = dataset.count()
    n_use = ntot if args.n <= 0 else min(args.n, ntot)
    rng = np.random.default_rng(SEED)
    idx_all = rng.choice(ntot, size=n_use, replace=False) if n_use < ntot else np.arange(ntot)
    arr = dataset.load_array()
    print(f"[nitido-v3] {n_use} imgs · epochs {start_epoch}..{args.epochs} · bs {args.batch} · "
          f"lr {args.lr} (warmup {WARMUP_STEPS}) · T={T} cosine · EMA 0.999 · {device.type}", flush=True)

    def batches(ix: np.ndarray):
        for i in range(0, len(ix), args.batch):
            b = ix[i : i + args.batch]
            t = torch.from_numpy(np.asarray(arr[b])).float().div_(255).permute(0, 3, 1, 2).contiguous()
            yield t.mul_(2.0).sub_(1.0).to(device)

    for ep in range(start_epoch, args.epochs + 1):
        rng.shuffle(idx_all)
        run, nb, t0 = 0.0, 0, time.time()
        for batch in batches(idx_all):
            gstep += 1
            # warmup de lr: sube linealmente hasta args.lr en los primeros WARMUP_STEPS pasos
            if gstep <= WARMUP_STEPS:
                for grp in opt.param_groups:
                    grp["lr"] = args.lr * gstep / WARMUP_STEPS
            b = batch.shape[0]
            t = torch.randint(0, T, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(batch)
            x_t = q_sample(sched, batch, t, noise)
            loss = F.mse_loss(model(x_t, t), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ema.update(model)
            run += loss.item()
            nb += 1
        last = run / max(nb, 1)
        print(f"[nitido-v3] epoch {ep}/{args.epochs} · loss {last:.5f} · {time.time() - t0:.0f}s", flush=True)

        if ep % args.save_every == 0 or ep == args.epochs:
            ema.copy_to(ema_model)
            _save_demo(ema_model, ep, last, args.lr)
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(), "epoch": ep, "gstep": gstep}, STATE_PATH)
            # rejilla de progreso (pocos pasos: orientativa, no la eval oficial)
            set_seed(SEED + ep)
            imgs, _ = sample_loop(ema_model, sched, 8, device, steps=40)
            quality.save_grid(imgs, quality.GRIDS_DIR / f"diffusion-nitido-v3-progress-ep{ep:03d}.png", ncol=8)
            set_seed(SEED)  # no perturbar la secuencia de entrenamiento más de lo justo
            print(f"[nitido-v3] === guardado epoch {ep} (demo + estado + grid) ===", flush=True)

    print("[nitido-v3] DONE", flush=True)


if __name__ == "__main__":
    main()
