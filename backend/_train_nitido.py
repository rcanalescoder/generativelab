"""Entrena la variante diffusion 'nitido' (64x64) con guardado periódico para poder
evaluar el avance sin esperar al final. Ejecutar con cwd=backend y el venv."""
import dataclasses
import time

import numpy as np
import torch
from torch.nn import functional as F

from app.data import dataset
from app.device import get_device
from app.models.diffusion import EMA, DiffusionSchedule, build_diffusion, q_sample
from app.seeding import set_seed
import app.services.diffusion_service as D

EPOCHS = 60
N = 18000
BS = 64
LR = 2e-4
SAVE_EVERY = 6

device = get_device()
set_seed(42)
model = build_diffusion("nitido").to(device); model.train()
sched = DiffusionSchedule(200, device, "cosine")
opt = torch.optim.Adam(model.parameters(), lr=LR)
ema = EMA(model, decay=0.999)
ema_model = build_diffusion("nitido").to(device); ema_model.eval()

arr = dataset.load_array(); ntot = dataset.count()
rng = np.random.default_rng(42)
idx_all = rng.choice(ntot, size=min(N, ntot), replace=False)

hp = dataclasses.replace(
    D.DiffusionHyperParams(epochs=EPOCHS, learning_rate=LR, schedule="cosine"), arch="nitido"
).sanitized()

def batches(ix):
    for i in range(0, len(ix), BS):
        b = ix[i:i + BS]
        t = torch.from_numpy(np.asarray(arr[b])).float().div_(255).permute(0, 3, 1, 2).contiguous()
        t = t.mul_(2.0).sub_(1.0)  # 64x64 nativo, [-1,1]
        yield t.to(device)

def save(ep, loss):
    ema.copy_to(ema_model); ema_model.eval()
    svc = D.diffusion_service
    svc.model = ema_model; svc.hp = hp; svc.seed = 42; svc.trained = True
    svc.loss_history = [{"epoch": ep, "loss": loss}]
    svc.save_checkpoint(D.demo_path("nitido"))
    print(f"[nitido] === checkpoint guardado en epoch {ep} (loss {loss:.5f}) ===", flush=True)

print(f"[nitido] entrenando 64x64 · {N} imgs · {EPOCHS} epochs · {device.type}", flush=True)
for ep in range(1, EPOCHS + 1):
    rng.shuffle(idx_all); run = 0.0; nb = 0; t0 = time.time()
    for batch in batches(idx_all):
        b = batch.shape[0]
        t = torch.randint(0, 200, (b,), device=device, dtype=torch.long)
        noise = torch.randn_like(batch)
        xt = q_sample(sched, batch, t, noise)
        loss = F.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); ema.update(model)
        run += loss.item(); nb += 1
    last = run / max(nb, 1)
    print(f"[nitido] epoch {ep}/{EPOCHS} · loss {last:.5f} · {time.time() - t0:.0f}s", flush=True)
    if ep % SAVE_EVERY == 0 or ep == EPOCHS:
        save(ep, last)
print("[nitido] DONE", flush=True)
