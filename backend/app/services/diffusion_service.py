"""Servicio de Diffusion (DDPM): estado, entrenamiento (generador de eventos), muestreo y
trayectoria. Sigue el mismo patrón hilo+cola SSE que el Autoencoder (CLAUDE.md §3.1–§3.3).

Un modelo de difusión GENERA partiendo de ruido puro y lo limpia paso a paso; no tiene
encoder. El entrenamiento es de *denoising*: para cada imagen tomamos un paso `t` aleatorio,
le añadimos ruido `ε` (forward `q_sample`) y entrenamos la UNet a predecir ese `ε` (MSE).

DOS VARIANTES elegibles (`build_diffusion(arch)`, `DIFF_ARCHS`), igual que en el Autoencoder:
  - "agil"   → UNet pequeña a 32×32 (los batches se bajan 64→32). Rápida; la de siempre.
  - "nitido" → UNet mayor a resolución NATIVA 64×64 (sin reescalado que emborrone) con
               auto-atención. Mucha más calidad a cambio de un muestreo bastante más lento.

RENDIMIENTO (MPS/CPU): "agil" trabaja a 32×32×3 con `T=200` y muestreo interactivo (DDIM, pocos
pasos). "nitido" trabaja a 64×64 y es notablemente más pesada. La resolución de trabajo la fija
SIEMPRE `model.img_size` (no una constante global): los batches y el ruido inicial se ajustan a
ella, y para mostrar se reescala a 64×64 (en "nitido" eso ya es la nativa, así que es identidad).
"""

from __future__ import annotations

import queue
import re
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
    DEFAULT_SCHEDULE,
    DIFF_ARCHS,
    TIMESTEPS,
    EMA,
    DiffusionSchedule,
    TimeUNet,
    build_diffusion,
    count_params,
    q_sample,
    sample_loop,
    sampling_timesteps,
)
from ..seeding import set_seed

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
# Checkpoint demo "legacy" (sin variantes): se trata como el demo de "agil".
LEGACY_DEMO_PATH = CKPT_DIR / "diffusion_demo.pt"


def demo_path(arch: str) -> Path:
    """Ruta del checkpoint demo por variante, p. ej. diffusion_agil_demo.pt."""
    return CKPT_DIR / f"diffusion_{arch}_demo.pt"


QUICK_N = 4000          # subconjunto para el modo "quick" (solo cambia el tamaño de datos)
DEMO_EPOCHS = 18        # checkpoint demo (modo full, acotado)
LOG_EVERY = 20          # steps entre eventos de progreso
EARLY_STOP_MIN_DELTA = 1e-4  # mejora mínima de pérdida por epoch para resetear la paciencia
PREVIEW_N = 6           # nº de caras del grid de vista previa por epoch
PREVIEW_STEPS = 50      # pasos de muestreo de la vista previa (rápida, con pesos EMA)
DEFAULT_SAMPLE_STEPS = 80  # pasos de muestreo por defecto de generate/trajectory (más calidad)
DISPLAY_SIZE = 64       # resolución a la que reescalamos para mostrar
# Decay de la EMA de pesos. 0.999 es el valor habitual; con entrenamientos cortos conviene
# bajarlo (0.995) para que la EMA "siga" antes a los pesos y no se quede en la inicialización.
EMA_DECAY = 0.999
EMA_DECAY_SHORT = 0.995
EMA_SHORT_EPOCHS = 15      # umbral de epochs por debajo del cual usamos el decay corto

# ---- config de los checkpoints demo (solo las usa el __main__ generador) -------------
# "agil": rápido de generar (subconjunto, pocas epochs), pensado como demo "de siempre".
AGIL_DEMO_N = 20000        # subconjunto de "agil" (modo quick)
AGIL_DEMO_EPOCHS = 30
# "nitido": config de CALIDAD pensada para entrenar LARGO (la lanza el humano y la deja correr).
# Sube `NITIDO_EPOCHS` y/o `NITIDO_DEMO_N` a placer; con el dataset completo, pon NITIDO_DEMO_N=None.
NITIDO_DEMO_N = 40000      # subconjunto grande (o None para usar el dataset completo)
NITIDO_EPOCHS = 120        # epochs altos: la calidad de difusión necesita entrenamiento largo
DEMO_LR = 2e-4             # lr de ambos demos


def _remap_legacy_agil_state(sd: dict) -> dict:
    """Adapta un state_dict legacy (UNet vieja con `down1/down2/up1/up2`) al naming actual
    (`downs.0/downs.1/ups.0/ups.1`). El bottleneck (`mid1/mid2`) y el resto no cambian.

    La arquitectura de "agil" es idéntica a la vieja, solo cambió el nombre de los módulos al
    pasar a `nn.ModuleList`. Así el viejo `diffusion_demo.pt` sigue cargando como demo de "agil".
    """
    out: dict = {}
    for k, v in sd.items():
        md = re.match(r"^down(\d+)(\..*)$", k)
        if md:
            out[f"downs.{int(md.group(1)) - 1}{md.group(2)}"] = v
            continue
        mu = re.match(r"^up(\d+)(\..*)$", k)
        if mu:
            out[f"ups.{int(mu.group(1)) - 1}{mu.group(2)}"] = v
            continue
        out[k] = v
    return out


@dataclass
class DiffusionHyperParams:
    """Defaults = la receta ÓPTIMA del estudio v3 (KID 19,4, «Lista de Mejoras.md» §5):
    variante nitido (64×64 con atención) y 28 epochs. «agil» queda a un clic para
    interactividad máxima (muestrea mucho más rápido)."""

    learning_rate: float = 2e-4
    epochs: int = 28
    batch_size: int = 128
    timesteps: int = TIMESTEPS
    schedule: str = DEFAULT_SCHEDULE  # "cosine" (def.) | "linear"
    arch: str = "nitido"             # "agil" | "nitido"

    def sanitized(self) -> "DiffusionHyperParams":
        return DiffusionHyperParams(
            learning_rate=float(max(1e-5, min(1e-2, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            batch_size=int(max(16, min(512, self.batch_size))),
            timesteps=int(max(50, min(1000, self.timesteps))),
            schedule="linear" if str(self.schedule).lower() == "linear" else "cosine",
            arch=self.arch if self.arch in DIFF_ARCHS else "agil",
        )


class DiffusionService:
    """Mantiene la UNet de difusión y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        # `self.model` aloja SIEMPRE los pesos que se usan para muestrear. Tras entrenar o
        # cargar un checkpoint, son los pesos EMA (suavizados), que dan mejores muestras.
        self.model: TimeUNet | None = None
        self.hp = DiffusionHyperParams()
        self.seed = 42
        self.trained = False
        self.training = False
        self.loss_history: list[dict] = []
        self._cancel = False
        self.sched = DiffusionSchedule(self.hp.timesteps, self.device, self.hp.schedule)

    # ---------- estado ----------
    def status(self) -> dict:
        return {
            "model": "diffusion",
            "trained": self.trained,
            "training": self.training,
            "seed": self.seed,
            "device": self.device.type,
            "timesteps": self.hp.timesteps,
            "arch": self.hp.arch,
            "archs": list(DIFF_ARCHS),
            "hyperparams": asdict(self.hp),
            "num_params": count_params(self.model) if self.model is not None else 0,
            "loss_history": self.loss_history,
        }

    def cancel(self) -> None:
        self._cancel = True

    def _ensure_schedule(self, timesteps: int, schedule: str | None = None) -> None:
        sched_type = schedule or self.hp.schedule
        if self.sched.timesteps != timesteps or self.sched.schedule != sched_type:
            self.sched = DiffusionSchedule(timesteps, self.device, sched_type)
        else:
            self.sched.to(self.device)

    # ---------- datos ----------
    def _iter_batches(
        self, indices: np.ndarray, batch_size: int, img_size: int
    ) -> Iterator[torch.Tensor]:
        """Carga batches del dataset 64×64, los ajusta a `img_size` y los lleva a [-1,1].

        Para "nitido" (img_size=64) el redimensionado es prácticamente la identidad (la nativa
        del dataset); para "agil" (img_size=32) baja la resolución como hasta ahora.
        """
        arr = dataset.load_array()
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            chunk = np.asarray(arr[idx])  # (B,64,64,3) uint8
            t = torch.from_numpy(chunk).float().div_(255).permute(0, 3, 1, 2).contiguous()
            if t.shape[-1] != img_size:
                t = F.interpolate(t, size=img_size, mode="bilinear", align_corners=False)
            t = t.mul_(2.0).sub_(1.0)  # [0,1] -> [-1,1]
            yield t.to(self.device)

    def _preview_grid(self, model: TimeUNet, n: int = PREVIEW_N) -> list[str]:
        """Genera un grid pequeño de muestras (pocos pasos) para la vista previa del epoch."""
        imgs, _ = sample_loop(model, self.sched, n, self.device, steps=PREVIEW_STEPS)
        return [self._encode(imgs[k]) for k in range(imgs.shape[0])]

    def _encode(self, img01: torch.Tensor) -> str:
        """Tensor CHW [0,1] (a la resolución del modelo) → PNG base64 reescalado a 64×64.

        En "nitido" la imagen ya llega a 64×64 (su resolución nativa) → reescalado identidad;
        en "agil" sube de 32×32 a 64×64 para mostrar.
        """
        if img01.shape[-1] != DISPLAY_SIZE:
            img01 = F.interpolate(
                img01.unsqueeze(0), size=DISPLAY_SIZE, mode="bilinear", align_corners=False
            )[0]
        return imaging.tensor_to_b64(img01)

    # ---------- entrenamiento ----------
    def iter_train(
        self, mode: str, hp: DiffusionHyperParams, seed: int, early_stop: bool = True, patience: int = 5
    ) -> Iterator[dict]:
        """Entrena el denoising emitiendo eventos {type, epoch, step, loss, preview?}.

        Adopta el modelo en memoria al terminar; NO guarda checkpoint (igual que el AE: la
        app no toca el demo; el modelo entrenado vive durante esta sesión del servidor).

        El número de epochs del slider se respeta siempre; "quick" solo reduce el tamaño del
        subconjunto. Con early_stop, para antes si la pérdida no mejora durante `patience` epochs.
        """
        if self.training:
            yield {"type": "error", "message": "Ya hay un entrenamiento en curso"}
            return
        self.training = True
        self._cancel = False
        try:
            hp = hp.sanitized()
            set_seed(seed)
            self._ensure_schedule(hp.timesteps, hp.schedule)
            n_total = dataset.count()
            quick = mode == "quick"
            n_use = min(QUICK_N, n_total) if quick else n_total
            epochs = hp.epochs

            # Construye la UNet de la variante elegida (define la resolución de trabajo).
            model = build_diffusion(hp.arch).to(self.device)
            model.train()
            img_size = model.img_size
            opt = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

            # EMA de pesos: decay corto si el entrenamiento es breve (para que la EMA se
            # "caliente" a tiempo y no se quede atascada cerca de la inicialización).
            ema_decay = EMA_DECAY if epochs >= EMA_SHORT_EPOCHS else EMA_DECAY_SHORT
            ema = EMA(model, decay=ema_decay)
            # modelo "shadow" reutilizable (MISMA variante) donde volcamos los pesos EMA para muestrear.
            ema_model = build_diffusion(hp.arch).to(self.device)
            ema_model.eval()

            rng = np.random.default_rng(seed)
            indices = (
                rng.choice(n_total, size=n_use, replace=False)
                if n_use < n_total
                else np.arange(n_total)
            )

            history: list[dict] = []
            global_step = 0
            best = float("inf")
            no_improve = 0
            stopped_early = False
            yield {"type": "start", "epochs": epochs, "n": int(n_use), "mode": mode,
                   "hyperparams": asdict(hp), "seed": seed}

            for epoch in range(1, epochs + 1):
                rng.shuffle(indices)
                running = torch.zeros((), device=self.device)
                nb = 0
                for batch in self._iter_batches(indices, hp.batch_size, img_size):
                    b = batch.shape[0]
                    t = torch.randint(0, hp.timesteps, (b,), device=self.device, dtype=torch.long)
                    noise = torch.randn_like(batch)
                    x_t = q_sample(self.sched, batch, t, noise)
                    eps_pred = model(x_t, t)
                    loss = F.mse_loss(eps_pred, noise)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    ema.update(model)  # actualizar la EMA tras cada paso del optimizador
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
                # vista previa SIEMPRE con los pesos EMA (mejor calidad que los pesos crudos).
                ema.copy_to(ema_model)
                yield {"type": "epoch", "epoch": epoch, "epochs": epochs, "step": global_step,
                       "loss": epoch_loss, "preview": self._preview_grid(ema_model)}
                if early_stop:
                    if epoch_loss < best - EARLY_STOP_MIN_DELTA:
                        best = epoch_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            stopped_early = True
                            break

            # finalizar: adoptar los pesos EMA (no los crudos) como el modelo de muestreo.
            # Es lo que da las mejores muestras y lo que guardará el checkpoint demo.
            ema.copy_to(ema_model)
            ema_model.eval()
            self.model = ema_model
            self.hp = hp
            self.seed = seed
            self.trained = True
            self.loss_history = history
            yield {"type": "done", "epochs": epochs, "epochs_run": len(history),
                   "stopped_early": stopped_early,
                   "loss": history[-1]["loss"] if history else None, "loss_history": history}
        finally:
            self.training = False

    def start_training(
        self, mode: str, hp: DiffusionHyperParams, seed: int, early_stop: bool = True
    ) -> "queue.Queue":
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
                for ev in self.iter_train(mode, hp, seed, early_stop=early_stop):
                    q.put(ev)
            except Exception as exc:  # pragma: no cover - defensivo
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, name="diffusion-train", daemon=True).start()
        return q

    # ---------- generación ----------
    def generate(self, n: int, seed: int, steps: int | None = None) -> list[str]:
        """Muestrea `n` imágenes desde ruido puro. `steps` = nº de pasos (submuestreo de T)."""
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        n = int(max(1, min(16, n)))
        steps = DEFAULT_SAMPLE_STEPS if steps is None else steps
        steps = int(max(2, min(self.hp.timesteps, steps)))
        set_seed(seed)
        self._ensure_schedule(self.hp.timesteps)
        imgs, _ = sample_loop(self.model, self.sched, n, self.device, steps=steps)
        return [self._encode(imgs[k]) for k in range(imgs.shape[0])]

    def trajectory(self, seed: int, steps: int | None = None, snapshots: int = 8) -> list[dict]:
        """Genera UNA imagen y devuelve `snapshots` instantáneas a lo largo del muestreo.

        Visualiza el proceso de difusión: de ruido puro (paso 0 del muestreo) a cara (último).
        Cada frame trae su posición en la secuencia y el `t` del schedule correspondiente.
        """
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        steps = DEFAULT_SAMPLE_STEPS if steps is None else steps
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
    def save_checkpoint(self, path: Path | None = None) -> None:
        """Guarda el modelo en `diffusion_<arch>_demo.pt`. Solo lo usa el `__main__` del demo.

        `self.model` contiene los pesos EMA tras entrenar, así que el `state_dict` guardado
        es directamente la EMA: el demo muestrea con los pesos suavizados (mejor calidad).
        """
        if self.model is None:
            return
        path = path or demo_path(self.hp.arch)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hyperparams": asdict(self.hp),
                "arch": self.hp.arch,
                "seed": self.seed,
                "loss_history": self.loss_history,
                "version": 2,
            },
            path,
        )

    def _load_path(self, path: Path, arch: str) -> bool:
        """Carga un checkpoint de la variante `arch` desde una ruta concreta.

        Construye la UNet de la arch guardada y carga los pesos EMA (estricto). Tolera el
        checkpoint legacy `diffusion_demo.pt` (sin variantes) remapeando sus claves a "agil".
        Devuelve False (sin romper) si no existe o los pesos son incompatibles.
        """
        if not path.exists():
            return False
        try:
            ckpt = torch.load(path, map_location=self.device)
            hp = replace(DiffusionHyperParams(**ckpt["hyperparams"]), arch=arch).sanitized()
            model = build_diffusion(arch).to(self.device)
            sd = ckpt["state_dict"]
            # legacy "agil" (UNet vieja con down1/up1…): adapta el naming al actual.
            if arch == "agil" and not any(k.startswith("downs.") for k in sd):
                sd = _remap_legacy_agil_state(sd)
            model.load_state_dict(sd)  # estricto: detecta archs incompatibles
        except Exception as exc:  # checkpoint viejo/corrupto: tratar como "sin demo"
            print(f"[diffusion] checkpoint '{arch}' incompatible, se ignora ({exc}). "
                  f"Regenéralo con: python -m app.services.diffusion_service", flush=True)
            return False
        self.hp = hp
        self._ensure_schedule(self.hp.timesteps, self.hp.schedule)
        model.eval()
        self.model = model
        self.seed = ckpt.get("seed", 42)
        self.loss_history = ckpt.get("loss_history", [])
        self.trained = True
        return True

    def load_demo(self, arch: str) -> bool:
        """Carga el checkpoint demo de la variante `arch` (`diffusion_<arch>_demo.pt`).

        Devuelve True si se cargó. Compatibilidad: si se pide "agil" y solo existe el viejo
        `diffusion_demo.pt`, se usa ese. Lo usa la UI para cambiar de variante al instante.
        """
        if arch not in DIFF_ARCHS:
            return False
        path = demo_path(arch)
        if not path.exists() and arch == "agil" and LEGACY_DEMO_PATH.exists():
            path = LEGACY_DEMO_PATH
        return self._load_path(path, arch)

    def load_checkpoint(self, arch: str | None = None) -> bool:
        """Carga el demo de la variante `arch` (o la actual). Compatible con el viejo
        `diffusion_demo.pt` (se trata como "agil").

        Al arrancar el servidor se llama sin argumentos: intenta la variante actual y, si no
        hay, prueba el resto de variantes (v3): si solo existe el demo de "nitido", la app
        arranca con él en vez de quedarse "sin entrenar".
        """
        if arch is not None:
            return self.load_demo(arch)
        if self.load_demo(self.hp.arch):
            return True
        for other in DIFF_ARCHS:
            if other != self.hp.arch and self.load_demo(other):
                return True
        return False


# Singleton compartido por los routers.
diffusion_service = DiffusionService()


def _train_demo_variant(arch: str) -> None:
    """Entrena UNA variante con su config de demo y guarda `diffusion_<arch>_demo.pt`.

    Guarda los pesos EMA (mejor muestreo). Schedule cosine (T=200) + EMA en ambas. La loss de
    denoising (MSE de ε) NO mide calidad de muestra, así que no optimizamos con grid search:
    aplicamos buenas prácticas (EMA + cosine + entrenamiento largo). Desactivamos early_stop
    para gastar el presupuesto completo de epochs. Los checkpoints NO se versionan: cada
    usuario los regenera con este comando; esto NO afecta al botón "Rápido" de la app.
    """
    svc = DiffusionService()
    n_total = dataset.count()
    # `iter_train` usa el módulo-global QUICK_N como tope del subconjunto en modo "quick".
    # Lo fijamos aquí (por variante) para controlar exactamente cuántas imágenes se usan.
    if arch == "nitido":
        # Config de CALIDAD pensada para entrenar LARGO (dataset grande/completo, muchas epochs).
        epochs = NITIDO_EPOCHS
        if NITIDO_DEMO_N is None:
            mode, n_use = "full", n_total           # dataset completo
        else:
            mode, n_use = "quick", min(NITIDO_DEMO_N, n_total)
            globals()["QUICK_N"] = n_use
    else:
        # "agil": como hasta ahora (subconjunto, pocas epochs, rápido).
        epochs = AGIL_DEMO_EPOCHS
        mode, n_use = "quick", min(AGIL_DEMO_N, n_total)
        globals()["QUICK_N"] = n_use

    hp = DiffusionHyperParams(
        epochs=epochs, learning_rate=DEMO_LR, schedule="cosine", arch=arch
    )
    print(
        f"[diffusion] === variante '{arch}' · {epochs} epochs · n≈{n_use} · lr {DEMO_LR} · "
        f"cosine · T={TIMESTEPS} · EMA · {mode} en {svc.device.type} ===",
        flush=True,
    )
    for ev in svc.iter_train(mode, hp, seed=42, early_stop=False):
        if ev["type"] == "step":
            print(f"[diffusion]   [{arch}] step {ev['step']} · loss {ev['loss']:.5f}", flush=True)
        elif ev["type"] == "epoch":
            print(f"[diffusion]  [{arch}] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}", flush=True)
        elif ev["type"] == "done":
            final = ev["loss"]
            msg = f"loss final {final:.5f}" if final is not None else "sin loss"
            print(f"[diffusion]  [{arch}] listo · {msg}", flush=True)
    path = demo_path(arch)
    svc.save_checkpoint(path)
    params = count_params(svc.model) if svc.model is not None else 0
    print(
        f"[diffusion]  [{arch}] checkpoint (pesos EMA, {params / 1e6:.2f}M params) guardado en {path}",
        flush=True,
    )


if __name__ == "__main__":
    # Genera los checkpoints demo de AMBAS variantes:
    #     python -m app.services.diffusion_service
    #
    # "agil" es rápido; "nitido" está pensado para entrenar LARGO (sube NITIDO_EPOCHS /
    # NITIDO_DEMO_N arriba). Cada variante guarda su propio diffusion_<arch>_demo.pt.
    for _arch in DIFF_ARCHS:
        _train_demo_variant(_arch)
    print("[diffusion] === demos de todas las variantes generados ===", flush=True)
