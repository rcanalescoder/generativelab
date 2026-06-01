"""Servicio del GAN (DCGAN): estado, entrenamiento adversarial (generador de eventos),
checkpoint y muestreo. Mismo patrón hilo+cola para SSE que el autoencoder (CLAUDE.md §3.1).

Un GAN no reconstruye ni tiene vecinos: su esencia es **generar** desde ruido z y el juego
entre el generador (G) y el discriminador (D). El entrenamiento emite el preview de un z
fijo en cada epoch para ver cómo el generador "aprende a dibujar" caras.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from .. import imaging
from ..data import dataset
from ..device import get_device
from ..models.gan import (
    Discriminator,
    Generator,
    count_params,
    init_weights,
    to_image_range,
)
from ..seeding import set_seed

DEMO_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "gan_demo.pt"

QUICK_N = 6000           # subconjunto para el modo "quick"
QUICK_MAX_EPOCHS = 6
DEMO_N = 15000           # subconjunto para el checkpoint demo (acota el tiempo)
DEMO_EPOCHS = 25
LOG_EVERY = 25           # steps entre eventos de progreso
PREVIEW_N = 6            # nº de caras en el grid de vista previa por epoch
DEMO_BATCH = 128


@dataclass
class GANHyperParams:
    z_dim: int = 100
    learning_rate: float = 2e-4
    epochs: int = 30
    batch_size: int = 128

    def sanitized(self) -> "GANHyperParams":
        return GANHyperParams(
            z_dim=int(max(16, min(256, self.z_dim))),
            learning_rate=float(max(1e-5, min(1e-2, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            batch_size=int(max(16, min(256, self.batch_size))),
        )


class GANService:
    """Mantiene el generador + discriminador y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.generator: Generator | None = None
        self.discriminator: Discriminator | None = None
        self.hp = GANHyperParams()
        self.seed = 42
        self.trained = False
        self.outdated = False
        self.training = False
        # historial: [{epoch, g_loss, d_loss}]
        self.loss_history: list[dict] = []
        self._cancel = False

    # ---------- estado ----------
    def status(self) -> dict:
        ng = count_params(self.generator) if self.generator is not None else 0
        nd = count_params(self.discriminator) if self.discriminator is not None else 0
        return {
            "model": "gan",
            "trained": self.trained,
            "outdated": self.outdated,
            "training": self.training,
            "seed": self.seed,
            "device": self.device.type,
            "z_dim": self.hp.z_dim,
            "hyperparams": asdict(self.hp),
            # num_params = G + D (para la UI). Desglose por si interesa.
            "num_params": ng + nd,
            "num_params_g": ng,
            "num_params_d": nd,
            "loss_history": self.loss_history,
        }

    def set_hyperparams(self, partial: dict) -> dict:
        new = replace(self.hp, **{k: v for k, v in partial.items() if hasattr(self.hp, k)}).sanitized()
        if self.trained and asdict(new) != asdict(self.hp):
            self.outdated = True
        self.hp = new
        return self.status()

    def cancel(self) -> None:
        self._cancel = True

    # ---------- datos ----------
    def _iter_batches(self, indices: np.ndarray, batch_size: int) -> Iterator[torch.Tensor]:
        """Lotes de imágenes en [-1,1] (el rango de salida de la tanh del generador)."""
        arr = dataset.load_array()
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            chunk = np.asarray(arr[idx])  # (B,64,64,3) uint8
            t = torch.from_numpy(chunk).float().div_(127.5).sub_(1.0)  # [0,255] -> [-1,1]
            t = t.permute(0, 3, 1, 2).contiguous()
            yield t.to(self.device)

    def _preview(self, generator: Generator, fixed_z: torch.Tensor) -> list[str]:
        """Grid pequeño de caras generadas desde un z fijo (para ver el progreso)."""
        was_training = generator.training
        generator.eval()
        with torch.no_grad():
            imgs = to_image_range(generator(fixed_z))
        if was_training:
            generator.train()
        return [imaging.tensor_to_b64(imgs[k]) for k in range(imgs.shape[0])]

    # ---------- entrenamiento adversarial ----------
    def iter_train(
        self, mode: str, hp: GANHyperParams, seed: int, n_override: int | None = None
    ) -> Iterator[dict]:
        """Entrena emitiendo eventos {type, epoch, g_loss, d_loss, preview?}.

        Juego estándar (BCE con logits):
          - D maximiza  log D(x_real) + log(1 − D(G(z)))   → minimiza BCE con etiquetas 1/0.
          - G minimiza  log(1 − D(G(z)))  ⇒ en la práctica MAXIMIZA log D(G(z))
            (truco "non-saturating": etiquetas reales para las muestras falsas).

        `n_override` acota el subconjunto de imágenes (lo usa el demo de `__main__` para
        que el tiempo de entrenamiento sea razonable sin tocar el dataset global).
        """
        if self.training:
            yield {"type": "error", "message": "Ya hay un entrenamiento en curso"}
            return
        self.training = True
        self._cancel = False
        try:
            hp = hp.sanitized()
            set_seed(seed)
            n_total = dataset.count()
            quick = mode == "quick"
            n_use = min(QUICK_N, n_total) if quick else n_total
            if n_override is not None:
                n_use = min(n_override, n_total)
            epochs = min(hp.epochs, QUICK_MAX_EPOCHS) if quick else hp.epochs

            generator = Generator(hp.z_dim).to(self.device)
            discriminator = Discriminator().to(self.device)
            generator.apply(init_weights)
            discriminator.apply(init_weights)
            generator.train()
            discriminator.train()

            betas = (0.5, 0.999)  # betas Adam recomendadas para DCGAN
            opt_g = torch.optim.Adam(generator.parameters(), lr=hp.learning_rate, betas=betas)
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=hp.learning_rate, betas=betas)
            criterion = nn.BCEWithLogitsLoss()

            # z fijo para la vista previa: misma "semilla visual" en todas las epochs,
            # así se ve cómo evolucionan exactamente las mismas caras.
            fixed_z = torch.randn(PREVIEW_N, hp.z_dim, device=self.device)

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
                run_g = torch.zeros((), device=self.device)
                run_d = torch.zeros((), device=self.device)
                nb = 0
                for real in self._iter_batches(indices, hp.batch_size):
                    bs = real.size(0)
                    real_labels = torch.ones(bs, device=self.device)
                    fake_labels = torch.zeros(bs, device=self.device)

                    # ----- (1) paso del discriminador -----
                    opt_d.zero_grad(set_to_none=True)
                    out_real = discriminator(real)
                    loss_real = criterion(out_real, real_labels)
                    z = torch.randn(bs, hp.z_dim, device=self.device)
                    fake = generator(z)
                    out_fake = discriminator(fake.detach())  # detach: no propaga a G
                    loss_fake = criterion(out_fake, fake_labels)
                    loss_d = loss_real + loss_fake
                    loss_d.backward()
                    opt_d.step()

                    # ----- (2) paso del generador -----
                    opt_g.zero_grad(set_to_none=True)
                    out_fake_g = discriminator(fake)  # reusar las falsas, ahora SÍ propaga a G
                    loss_g = criterion(out_fake_g, real_labels)  # non-saturating
                    loss_g.backward()
                    opt_g.step()

                    run_g += loss_g.detach()
                    run_d += loss_d.detach()
                    nb += 1
                    global_step += 1
                    if global_step % LOG_EVERY == 0:
                        yield {"type": "step", "epoch": epoch, "step": global_step,
                               "g_loss": float((run_g / nb).item()),
                               "d_loss": float((run_d / nb).item())}
                    if self._cancel:
                        yield {"type": "cancelled", "epoch": epoch, "step": global_step}
                        return

                g_loss = float((run_g / max(nb, 1)).item())
                d_loss = float((run_d / max(nb, 1)).item())
                history.append({"epoch": epoch, "g_loss": g_loss, "d_loss": d_loss})
                yield {"type": "epoch", "epoch": epoch, "epochs": epochs, "step": global_step,
                       "g_loss": g_loss, "d_loss": d_loss,
                       "preview": self._preview(generator, fixed_z)}

            # finalizar: adoptar el modelo entrenado (en memoria; NO toca el checkpoint demo)
            generator.eval()
            discriminator.eval()
            self.generator = generator
            self.discriminator = discriminator
            self.hp = hp
            self.seed = seed
            self.trained = True
            self.outdated = False
            self.loss_history = history
            yield {"type": "done", "epochs": epochs,
                   "g_loss": history[-1]["g_loss"] if history else None,
                   "d_loss": history[-1]["d_loss"] if history else None,
                   "loss_history": history}
        finally:
            self.training = False

    def start_training(self, mode: str, hp: GANHyperParams, seed: int) -> "queue.Queue":
        """Arranca el entrenamiento en un hilo propio y devuelve una cola de eventos.

        Igual que el AE: desacopla el entrenamiento de la conexión del cliente. El None
        final es el centinela de fin de stream.
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

        threading.Thread(target=worker, name="gan-train", daemon=True).start()
        return q

    # ---------- generación / interpolación ----------
    def _require_generator(self) -> Generator:
        if self.generator is None:
            raise RuntimeError("modelo no entrenado")
        return self.generator

    def generate(self, n: int, seed: int) -> list[str]:
        """Genera n imágenes desde z ~ N(0,1). Devuelve PNG base64 (data URL)."""
        gen = self._require_generator()
        n = int(max(1, min(64, n)))
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        z = torch.randn(n, self.hp.z_dim, generator=g).to(self.device)
        gen.eval()
        with torch.no_grad():
            imgs = to_image_range(gen(z))
        return [imaging.tensor_to_b64(imgs[k]) for k in range(n)]

    def interpolate(self, steps: int, seed: int) -> dict:
        """Interpola en el espacio de ruido z entre dos vectores aleatorios (por semilla).

        Es lo análogo a la interpolación del AE, pero en el espacio del prior: como el GAN
        no tiene encoder, no se parte de imágenes del dataset sino de dos puntos de ruido.
        """
        gen = self._require_generator()
        steps = int(max(2, min(16, steps)))
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        za = torch.randn(1, self.hp.z_dim, generator=g)
        zb = torch.randn(1, self.hp.z_dim, generator=g)
        gen.eval()
        frames: list[dict] = []
        with torch.no_grad():
            for s in range(steps):
                alpha = s / (steps - 1)
                z = ((1 - alpha) * za + alpha * zb).to(self.device)
                img = to_image_range(gen(z))[0]
                frames.append({"alpha": round(alpha, 3), "image": imaging.tensor_to_b64(img)})
        return {"steps": steps, "seed": int(seed), "frames": frames}

    # ---------- checkpoint ----------
    def save_checkpoint(self, path: Path = DEMO_PATH) -> None:
        """Guarda G + D. Solo lo usa el script generador del demo (`__main__`)."""
        if self.generator is None or self.discriminator is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "hyperparams": asdict(self.hp),
                "seed": self.seed,
                "loss_history": self.loss_history,
                "version": 1,
            },
            path,
        )

    def load_checkpoint(self) -> bool:
        """Carga el generador (y el discriminador) para inferencia."""
        if not DEMO_PATH.exists():
            return False
        ckpt = torch.load(DEMO_PATH, map_location=self.device)
        self.hp = GANHyperParams(**ckpt["hyperparams"]).sanitized()
        generator = Generator(self.hp.z_dim).to(self.device)
        generator.load_state_dict(ckpt["generator"])
        generator.eval()
        self.generator = generator
        discriminator = Discriminator().to(self.device)
        if "discriminator" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator"])
        discriminator.eval()
        self.discriminator = discriminator
        self.seed = ckpt.get("seed", 42)
        self.loss_history = ckpt.get("loss_history", [])
        self.trained = True
        self.outdated = False
        return True


# Singleton compartido por los routers.
gan_service = GANService()


if __name__ == "__main__":
    # Genera el checkpoint demo:  python -m app.services.gan_service
    # Config acotada para que el tiempo sea razonable en un Mac: subconjunto de DEMO_N
    # imágenes y DEMO_EPOCHS epochs. La calidad será tosca (es educativo); ver "riesgos".
    import time

    svc = GANService()
    print(
        f"[gan] entrenando checkpoint demo en {svc.device.type} · "
        f"subconjunto {DEMO_N} · {DEMO_EPOCHS} epochs · z_dim=100 · lr=2e-4 · Adam(0.5,0.999)…",
        flush=True,
    )
    hp = GANHyperParams(z_dim=100, learning_rate=2e-4, epochs=DEMO_EPOCHS, batch_size=DEMO_BATCH)
    t0 = time.time()
    # modo "full" pero acotado a DEMO_N imágenes vía n_override (sin tocar el dataset global).
    for ev in svc.iter_train("full", hp, seed=42, n_override=DEMO_N):
        if ev["type"] == "epoch":
            print(
                f"[gan]   epoch {ev['epoch']:>2}/{ev['epochs']} · "
                f"g_loss {ev['g_loss']:.4f} · d_loss {ev['d_loss']:.4f}",
                flush=True,
            )
        elif ev["type"] == "done":
            print(
                f"[gan] listo · g_loss {ev['g_loss']:.4f} · d_loss {ev['d_loss']:.4f} · "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
    svc.save_checkpoint(DEMO_PATH)
    print(f"[gan] checkpoint demo guardado en {DEMO_PATH}", flush=True)
