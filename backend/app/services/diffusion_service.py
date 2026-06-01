"""Servicio de Diffusion (DDPM): estado, entrenamiento (generador de eventos), muestreo y
trayectoria. Sigue el mismo patrón hilo+cola SSE que el Autoencoder (CLAUDE.md §3.1–§3.3).

Un modelo de difusión GENERA partiendo de ruido puro y lo limpia paso a paso; no tiene
encoder. El entrenamiento es de *denoising*: para cada imagen tomamos un paso `t` aleatorio,
le añadimos ruido `ε` (forward `q_sample`) y entrenamos la UNet a predecir ese `ε` (MSE).

RENDIMIENTO (MPS/CPU): trabajamos a 32×32×3 (downsample del dataset 64→32 al cargar batches),
UNet pequeña y `T=200`. El muestreo submuestrea pasos (p. ej. 50) para ser interactivo. Las
imágenes se reescalan a 64×64 para mostrar. La calidad es modesta a propósito: es educativo.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.nn import functional as F

from .. import imaging
from ..data import dataset
from ..device import get_device
from ..models.diffusion import (
    IMG_SIZE,
    TIMESTEPS,
    DiffusionSchedule,
    TimeUNet,
    count_params,
    q_sample,
    sample_loop,
    sampling_timesteps,
)
from ..seeding import set_seed

DEMO_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "diffusion_demo.pt"

QUICK_N = 4000          # subconjunto para el modo "quick"
QUICK_MAX_EPOCHS = 8
DEMO_EPOCHS = 18        # checkpoint demo (modo full, acotado)
LOG_EVERY = 20          # steps entre eventos de progreso
PREVIEW_N = 6           # nº de caras del grid de vista previa por epoch
PREVIEW_STEPS = 40      # pasos de muestreo de la vista previa (rápida)
DISPLAY_SIZE = 64       # resolución a la que reescalamos para mostrar


@dataclass
class DiffusionHyperParams:
    learning_rate: float = 2e-4
    epochs: int = 30
    batch_size: int = 128
    timesteps: int = TIMESTEPS

    def sanitized(self) -> "DiffusionHyperParams":
        return DiffusionHyperParams(
            learning_rate=float(max(1e-5, min(1e-2, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            batch_size=int(max(16, min(512, self.batch_size))),
            timesteps=int(max(50, min(1000, self.timesteps))),
        )


class DiffusionService:
    """Mantiene la UNet de difusión y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.model: TimeUNet | None = None
        self.hp = DiffusionHyperParams()
        self.seed = 42
        self.trained = False
        self.training = False
        self.loss_history: list[dict] = []
        self._cancel = False
        self.sched = DiffusionSchedule(self.hp.timesteps, self.device)

    # ---------- estado ----------
    def status(self) -> dict:
        return {
            "model": "diffusion",
            "trained": self.trained,
            "training": self.training,
            "seed": self.seed,
            "device": self.device.type,
            "timesteps": self.hp.timesteps,
            "hyperparams": asdict(self.hp),
            "num_params": count_params(self.model) if self.model is not None else 0,
            "loss_history": self.loss_history,
        }

    def cancel(self) -> None:
        self._cancel = True

    def _ensure_schedule(self, timesteps: int) -> None:
        if self.sched.timesteps != timesteps:
            self.sched = DiffusionSchedule(timesteps, self.device)
        else:
            self.sched.to(self.device)

    # ---------- datos ----------
    def _iter_batches(self, indices: np.ndarray, batch_size: int) -> Iterator[torch.Tensor]:
        """Carga batches del dataset 64×64, los baja a 32×32 y los lleva a [-1,1]."""
        arr = dataset.load_array()
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            chunk = np.asarray(arr[idx])  # (B,64,64,3) uint8
            t = torch.from_numpy(chunk).float().div_(255).permute(0, 3, 1, 2).contiguous()
            t = F.interpolate(t, size=IMG_SIZE, mode="bilinear", align_corners=False)
            t = t.mul_(2.0).sub_(1.0)  # [0,1] -> [-1,1]
            yield t.to(self.device)

    def _preview_grid(self, model: TimeUNet, n: int = PREVIEW_N) -> list[str]:
        """Genera un grid pequeño de muestras (pocos pasos) para la vista previa del epoch."""
        imgs, _ = sample_loop(model, self.sched, n, self.device, steps=PREVIEW_STEPS)
        return [self._encode(imgs[k]) for k in range(imgs.shape[0])]

    def _encode(self, img01: torch.Tensor) -> str:
        """Tensor CHW [0,1] (32×32) → PNG base64 reescalado a 64×64 para mostrar."""
        up = F.interpolate(img01.unsqueeze(0), size=DISPLAY_SIZE, mode="bilinear", align_corners=False)[0]
        return imaging.tensor_to_b64(up)

    # ---------- entrenamiento ----------
    def iter_train(self, mode: str, hp: DiffusionHyperParams, seed: int) -> Iterator[dict]:
        """Entrena el denoising emitiendo eventos {type, epoch, step, loss, preview?}.

        Adopta el modelo en memoria al terminar; NO guarda checkpoint (igual que el AE: la
        app no toca el demo; el modelo entrenado vive durante esta sesión del servidor).
        """
        if self.training:
            yield {"type": "error", "message": "Ya hay un entrenamiento en curso"}
            return
        self.training = True
        self._cancel = False
        try:
            hp = hp.sanitized()
            set_seed(seed)
            self._ensure_schedule(hp.timesteps)
            n_total = dataset.count()
            quick = mode == "quick"
            n_use = min(QUICK_N, n_total) if quick else n_total
            epochs = min(hp.epochs, QUICK_MAX_EPOCHS) if quick else hp.epochs

            model = TimeUNet().to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

            rng = np.random.default_rng(seed)
            indices = (
                rng.choice(n_total, size=n_use, replace=False)
                if n_use < n_total
                else np.arange(n_total)
            )

            history: list[dict] = []
            global_step = 0
            yield {"type": "start", "epochs": epochs, "n": int(n_use), "mode": mode,
                   "hyperparams": asdict(hp), "seed": seed}

            for epoch in range(1, epochs + 1):
                rng.shuffle(indices)
                running = torch.zeros((), device=self.device)
                nb = 0
                for batch in self._iter_batches(indices, hp.batch_size):
                    b = batch.shape[0]
                    t = torch.randint(0, hp.timesteps, (b,), device=self.device, dtype=torch.long)
                    noise = torch.randn_like(batch)
                    x_t = q_sample(self.sched, batch, t, noise)
                    eps_pred = model(x_t, t)
                    loss = F.mse_loss(eps_pred, noise)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    running += loss.detach()
                    nb += 1
                    global_step += 1
                    if global_step % LOG_EVERY == 0:
                        yield {"type": "step", "epoch": epoch, "step": global_step,
                               "loss": float((running / nb).item())}
                    if self._cancel:
                        yield {"type": "cancelled", "epoch": epoch, "step": global_step}
                        return
                epoch_loss = float((running / max(nb, 1)).item())
                history.append({"epoch": epoch, "loss": epoch_loss})
                yield {"type": "epoch", "epoch": epoch, "epochs": epochs, "step": global_step,
                       "loss": epoch_loss, "preview": self._preview_grid(model)}

            # finalizar: adoptar el modelo entrenado (sin tocar el checkpoint demo)
            model.eval()
            self.model = model
            self.hp = hp
            self.seed = seed
            self.trained = True
            self.loss_history = history
            yield {"type": "done", "epochs": epochs,
                   "loss": history[-1]["loss"] if history else None, "loss_history": history}
        finally:
            self.training = False

    def start_training(self, mode: str, hp: DiffusionHyperParams, seed: int) -> "queue.Queue":
        """Arranca el entrenamiento en un hilo propio y devuelve una cola de eventos.

        Desacopla el entrenamiento de la conexión del cliente: aunque el navegador se vaya,
        el hilo termina (o se cancela). El None final es el centinela de fin de stream.
        """
        q: "queue.Queue" = queue.Queue(maxsize=128)
        if self.training:
            q.put({"type": "error", "message": "Ya hay un entrenamiento en curso"})
            q.put(None)
            return q

        def worker() -> None:
            try:
                for ev in self.iter_train(mode, hp, seed):
                    q.put(ev)
            except Exception as exc:  # pragma: no cover - defensivo
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, name="diffusion-train", daemon=True).start()
        return q

    # ---------- generación ----------
    def generate(self, n: int, seed: int, steps: int) -> list[str]:
        """Muestrea `n` imágenes desde ruido puro. `steps` = nº de pasos (submuestreo de T)."""
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        n = int(max(1, min(16, n)))
        steps = int(max(2, min(self.hp.timesteps, steps)))
        set_seed(seed)
        self._ensure_schedule(self.hp.timesteps)
        imgs, _ = sample_loop(self.model, self.sched, n, self.device, steps=steps)
        return [self._encode(imgs[k]) for k in range(imgs.shape[0])]

    def trajectory(self, seed: int, steps: int, snapshots: int) -> list[dict]:
        """Genera UNA imagen y devuelve `snapshots` instantáneas a lo largo del muestreo.

        Visualiza el proceso de difusión: de ruido puro (paso 0 del muestreo) a cara (último).
        Cada frame trae su posición en la secuencia y el `t` del schedule correspondiente.
        """
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        steps = int(max(2, min(self.hp.timesteps, steps)))
        snapshots = int(max(2, min(steps + 1, snapshots)))
        set_seed(seed)
        self._ensure_schedule(self.hp.timesteps)

        ts = sampling_timesteps(self.hp.timesteps, steps)   # índices de tiempo, descendente
        total_positions = len(ts) + 1                       # +1 por el ruido inicial (pos 0)
        # posiciones (en la secuencia de muestreo) en las que capturar, repartidas uniformemente
        positions = sorted({int(round(p)) for p in np.linspace(0, total_positions - 1, snapshots)})
        imgs, snaps = sample_loop(
            self.model, self.sched, 1, self.device, steps=steps, capture=positions
        )

        frames: list[dict] = []
        for pos, snap in zip(positions, snaps):
            # t del schedule en esa posición: pos 0 = ruido puro (t=T); pos k usa ts[k-1]
            t_val = self.hp.timesteps if pos == 0 else int(ts[min(pos - 1, len(ts) - 1)])
            frames.append({
                "position": int(pos),
                "total": int(total_positions),
                "t": t_val,
                "image": self._encode(snap[0]),
            })
        # asegurar que el último frame es la imagen final limpia
        if not frames or frames[-1]["t"] != 0:
            frames.append({
                "position": int(total_positions - 1),
                "total": int(total_positions),
                "t": 0,
                "image": self._encode(imgs[0]),
            })
        return frames

    # ---------- checkpoint ----------
    def save_checkpoint(self, path: Path = DEMO_PATH) -> None:
        """Guarda el modelo. Solo lo usa el script generador del demo (`__main__`)."""
        if self.model is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hyperparams": asdict(self.hp),
                "seed": self.seed,
                "loss_history": self.loss_history,
                "version": 1,
            },
            path,
        )

    def load_checkpoint(self) -> bool:
        if not DEMO_PATH.exists():
            return False
        ckpt = torch.load(DEMO_PATH, map_location=self.device)
        self.hp = DiffusionHyperParams(**ckpt["hyperparams"]).sanitized()
        self._ensure_schedule(self.hp.timesteps)
        model = TimeUNet().to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model
        self.seed = ckpt.get("seed", 42)
        self.loss_history = ckpt.get("loss_history", [])
        self.trained = True
        return True


# Singleton compartido por los routers.
diffusion_service = DiffusionService()


if __name__ == "__main__":
    # Genera el checkpoint demo:  python -m app.services.diffusion_service
    svc = DiffusionService()
    print(
        f"[diffusion] entrenando checkpoint demo (full, {DEMO_EPOCHS} epochs, "
        f"T={TIMESTEPS}) en {svc.device.type}…",
        flush=True,
    )
    for ev in svc.iter_train("full", DiffusionHyperParams(epochs=DEMO_EPOCHS), seed=42):
        if ev["type"] == "step":
            print(f"[diffusion]   step {ev['step']} · loss {ev['loss']:.5f}", flush=True)
        elif ev["type"] == "epoch":
            print(f"[diffusion]  epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}", flush=True)
        elif ev["type"] == "done":
            final = ev["loss"]
            print(f"[diffusion] listo · loss final {final:.5f}" if final is not None else "[diffusion] listo", flush=True)
    svc.save_checkpoint(DEMO_PATH)
    print(f"[diffusion] checkpoint demo guardado en {DEMO_PATH}", flush=True)
