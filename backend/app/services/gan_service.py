"""Servicio del GAN (DCGAN): estado, entrenamiento adversarial (generador de eventos),
checkpoint y muestreo. Mismo patrón hilo+cola para SSE que el autoencoder (CLAUDE.md §3.1).

Un GAN no reconstruye ni tiene vecinos: su esencia es **generar** desde ruido z y el juego
entre el generador (G) y el discriminador (D). El entrenamiento emite el preview de un z
fijo en cada epoch para ver cómo el generador "aprende a dibujar" caras.

Dos variantes de arquitectura ("basico"/"grande", ver `models/gan.py`) con checkpoints demo
separados (`gan_<arch>_demo.pt`), igual que el AE. El GAN no tiene métrica de reconstrucción:
la comparación entre variantes es **visual** (galería de caras generadas).
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
from ..models.diffusion import EMA
from ..models.gan import (
    GAN_ARCHS,
    build_gan,
    count_params,
    diff_augment,
    init_weights,
    to_image_range,
)
from ..seeding import set_seed

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
# Checkpoint demo "legacy" (sin variantes): se trata como el demo de "basico".
LEGACY_DEMO_PATH = CKPT_DIR / "gan_demo.pt"


def demo_path(arch: str) -> Path:
    """Ruta del checkpoint demo por variante, p. ej. gan_basico_demo.pt."""
    return CKPT_DIR / f"gan_{arch}_demo.pt"


QUICK_N = 6000           # subconjunto para el modo "quick" (solo cambia el tamaño de datos)
DEMO_N = 15000           # subconjunto para el checkpoint demo (acota el tiempo)
DEMO_EPOCHS = 25
LOG_EVERY = 25           # steps entre eventos de progreso
PREVIEW_N = 6            # nº de caras en el grid de vista previa por epoch
DEMO_BATCH = 128


@dataclass
class GANHyperParams:
    """Defaults = la receta ÓPTIMA del estudio v3 (KID 37,5, «Lista de Mejoras.md» §4):
    EMA + label smoothing + DiffAugment, lrs iguales 2e-4 y 60 epochs."""

    z_dim: int = 100
    learning_rate: float = 2e-4
    epochs: int = 60
    batch_size: int = 128
    arch: str = "basico"  # "basico" | "grande"
    # --- estabilizadores v3 (ver «Lista de Mejoras.md» §4) ---
    # lr_g/lr_d: lrs separadas para G y D. None → se usa learning_rate para ambas.
    # El A/B a presupuesto completo ganó con lrs IGUALES (2e-4); TTUR (lr_g<lr_d) queda
    # como opción explorable desde la UI.
    lr_g: float | None = 2e-4
    lr_d: float | None = 2e-4
    label_smooth: float = 0.9  # etiqueta de "real" para D (suaviza su confianza)
    ema: bool = True           # media móvil de los pesos de G (se muestrea con ella)
    diffaug: bool = True       # DiffAugment sobre TODO lo que ve D (reales y falsas)

    def _clamp_lr(self, v: float | None) -> float | None:
        return None if v is None else float(max(1e-5, min(1e-2, v)))

    def sanitized(self) -> "GANHyperParams":
        return GANHyperParams(
            z_dim=int(max(16, min(256, self.z_dim))),
            learning_rate=float(max(1e-5, min(1e-2, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            batch_size=int(max(16, min(256, self.batch_size))),
            arch=self.arch if self.arch in GAN_ARCHS else "basico",
            lr_g=self._clamp_lr(self.lr_g),
            lr_d=self._clamp_lr(self.lr_d),
            label_smooth=float(max(0.5, min(1.0, self.label_smooth))),
            ema=bool(self.ema),
            diffaug=bool(self.diffaug),
        )


class GANService:
    """Mantiene el generador + discriminador y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.generator: nn.Module | None = None
        self.discriminator: nn.Module | None = None
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
            "arch": self.hp.arch,
            "archs": list(GAN_ARCHS),
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

    def _preview(self, generator: nn.Module, fixed_z: torch.Tensor) -> list[str]:
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
            # El nº de epochs del slider se respeta siempre; "quick" solo reduce el subconjunto.
            epochs = hp.epochs

            generator, discriminator = build_gan(hp.z_dim, hp.arch)
            generator = generator.to(self.device)
            discriminator = discriminator.to(self.device)
            generator.apply(init_weights)
            discriminator.apply(init_weights)
            generator.train()
            discriminator.train()

            betas = (0.5, 0.999)  # betas Adam recomendadas para DCGAN
            # TTUR (v3): lrs separadas — D algo más rápido que G suele estabilizar el juego.
            lr_g = hp.lr_g if hp.lr_g is not None else hp.learning_rate
            lr_d = hp.lr_d if hp.lr_d is not None else hp.learning_rate
            opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
            opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)
            criterion = nn.BCEWithLogitsLoss()

            # EMA del generador (v3): se muestrea con la media móvil de los pesos, no con los
            # pesos "crudos" que oscilan con cada minibatch. Decay corto si el run es breve.
            ema = EMA(generator, decay=0.999 if hp.epochs >= 15 else 0.995) if hp.ema else None
            ema_g: nn.Module | None = None
            if ema is not None:
                ema_g = build_gan(hp.z_dim, hp.arch)[0].to(self.device)
                ema_g.eval()

            # DiffAugment (v3): D nunca ve imágenes sin aumentar (reales y falsas por igual).
            aug = (lambda t: diff_augment(t)) if hp.diffaug else (lambda t: t)

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
                    # label smoothing (v3): a D se le pide ~0.9 para las reales, no 1.0 —
                    # evita que se vuelva sobreconfiado y deje a G sin gradiente útil.
                    real_labels = torch.full((bs,), hp.label_smooth, device=self.device)
                    fake_labels = torch.zeros(bs, device=self.device)
                    ones = torch.ones(bs, device=self.device)

                    # ----- (1) paso del discriminador -----
                    opt_d.zero_grad(set_to_none=True)
                    out_real = discriminator(aug(real))
                    loss_real = criterion(out_real, real_labels)
                    z = torch.randn(bs, hp.z_dim, device=self.device)
                    fake = generator(z)
                    out_fake = discriminator(aug(fake.detach()))  # detach: no propaga a G
                    loss_fake = criterion(out_fake, fake_labels)
                    loss_d = loss_real + loss_fake
                    loss_d.backward()
                    opt_d.step()

                    # ----- (2) paso del generador -----
                    opt_g.zero_grad(set_to_none=True)
                    out_fake_g = discriminator(aug(fake))  # reusar las falsas, ahora SÍ propaga a G
                    loss_g = criterion(out_fake_g, ones)  # non-saturating (target 1.0, sin smooth)
                    loss_g.backward()
                    opt_g.step()
                    if ema is not None:
                        ema.update(generator)

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
                # vista previa con los pesos EMA si están activos (lo que se verá al muestrear)
                if ema is not None and ema_g is not None:
                    ema.copy_to(ema_g)
                    preview_model = ema_g
                else:
                    preview_model = generator
                yield {"type": "epoch", "epoch": epoch, "epochs": epochs, "step": global_step,
                       "g_loss": g_loss, "d_loss": d_loss,
                       "preview": self._preview(preview_model, fixed_z)}

            # finalizar: adoptar el modelo entrenado (en memoria; NO toca el checkpoint demo).
            # Con EMA activa, el generador final lleva los pesos SUAVIZADOS (mejor muestreo).
            if ema is not None:
                ema.copy_to(generator)
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
    def _require_generator(self) -> nn.Module:
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
    def save_checkpoint(self, path: Path | None = None) -> None:
        """Guarda G + D en `gan_<arch>_demo.pt`. Solo lo usa el `__main__` generador del demo."""
        if self.generator is None or self.discriminator is None:
            return
        path = path or demo_path(self.hp.arch)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "hyperparams": asdict(self.hp),
                "arch": self.hp.arch,
                "seed": self.seed,
                "loss_history": self.loss_history,
                "version": 2,
            },
            path,
        )

    def _load_path(self, path: Path) -> bool:
        """Carga un checkpoint desde una ruta concreta. Tolera arch desconocida (devuelve False)."""
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        hp = GANHyperParams(**ckpt["hyperparams"])
        # arch puede venir suelta (v2) o dentro de hyperparams; v1 (legacy) → "basico".
        arch = ckpt.get("arch", getattr(hp, "arch", "basico"))
        if arch not in GAN_ARCHS:
            return False  # checkpoint de una variante que ya no existe: no crashear
        hp = replace(hp, arch=arch).sanitized()
        try:
            generator, discriminator = build_gan(hp.z_dim, hp.arch)
            generator = generator.to(self.device)
            discriminator = discriminator.to(self.device)
            generator.load_state_dict(ckpt["generator"])
            if "discriminator" in ckpt:
                discriminator.load_state_dict(ckpt["discriminator"])
        except (RuntimeError, KeyError):
            return False  # incompatibilidad de pesos: degradar sin romper
        generator.eval()
        discriminator.eval()
        self.generator = generator
        self.discriminator = discriminator
        self.hp = hp
        self.seed = ckpt.get("seed", 42)
        self.loss_history = ckpt.get("loss_history", [])
        self.trained = True
        self.outdated = False
        return True

    def load_demo(self, arch: str) -> bool:
        """Carga el checkpoint demo de la variante `arch` (`gan_<arch>_demo.pt`).

        Devuelve True si se cargó. Compatibilidad: si se pide "basico" y solo existe el viejo
        `gan_demo.pt`, se usa ese. Lo usa la UI para cambiar de variante al instante.
        """
        if arch not in GAN_ARCHS:
            return False
        path = demo_path(arch)
        if not path.exists() and arch == "basico" and LEGACY_DEMO_PATH.exists():
            path = LEGACY_DEMO_PATH
        return self._load_path(path)

    def load_checkpoint(self, arch: str | None = None) -> bool:
        """Carga el demo de la variante `arch` (o la actual). Compatible con el viejo `gan_demo.pt`.

        Al arrancar el servidor se llama sin argumentos: intenta la variante actual y, si no hay,
        cae a "basico" (que cubre el legacy `gan_demo.pt`) para que la app siempre tenga un demo.
        """
        if arch is not None:
            return self.load_demo(arch)
        if self.load_demo(self.hp.arch):
            return True
        if self.hp.arch != "basico":
            return self.load_demo("basico")
        return False


# Singleton compartido por los routers.
gan_service = GANService()


if __name__ == "__main__":
    # Genera el checkpoint demo de LAS DOS variantes:  python -m app.services.gan_service
    # Cada una se entrena en modo full acotado (subconjunto DEMO_N · DEMO_EPOCHS epochs) y se
    # guarda en gan_<arch>_demo.pt. La calidad será tosca (es educativo). El GAN no tiene
    # métrica de reconstrucción: la comparación entre variantes es visual (galería).
    import time

    summary: list[dict] = []
    for arch in GAN_ARCHS:
        svc = GANService()
        ng = count_params(build_gan(100, arch)[0])
        nd = count_params(build_gan(100, arch)[1])
        print(
            f"[gan] === variante '{arch}' · {arch} · subconjunto {DEMO_N} · {DEMO_EPOCHS} epochs "
            f"en {svc.device.type} · z_dim=100 · lr=2e-4 · Adam(0.5,0.999) · "
            f"G+D≈{(ng + nd) / 1e6:.1f}M params ===",
            flush=True,
        )
        hp = GANHyperParams(
            z_dim=100, learning_rate=2e-4, epochs=DEMO_EPOCHS, batch_size=DEMO_BATCH, arch=arch
        )
        t0 = time.time()
        # modo "full" pero acotado a DEMO_N imágenes vía n_override (sin tocar el dataset global).
        for ev in svc.iter_train("full", hp, seed=42, n_override=DEMO_N):
            if ev["type"] == "epoch":
                print(
                    f"[gan]   [{arch}] epoch {ev['epoch']:>2}/{ev['epochs']} · "
                    f"g_loss {ev['g_loss']:.4f} · d_loss {ev['d_loss']:.4f}",
                    flush=True,
                )
            elif ev["type"] == "done":
                print(
                    f"[gan]   [{arch}] listo · g_loss {ev['g_loss']:.4f} · "
                    f"d_loss {ev['d_loss']:.4f} · {time.time() - t0:.0f}s",
                    flush=True,
                )
        path = demo_path(arch)
        svc.save_checkpoint(path)
        print(f"[gan]   [{arch}] checkpoint demo guardado en {path}", flush=True)
        summary.append(
            {
                "arch": arch,
                "params": ng + nd,
                "g_loss": svc.loss_history[-1]["g_loss"] if svc.loss_history else None,
                "d_loss": svc.loss_history[-1]["d_loss"] if svc.loss_history else None,
            }
        )

    print("\n[gan] === Resumen de variantes (la comparación real es VISUAL) ===", flush=True)
    print(f"[gan] {'arch':8s} {'params':>10s} {'g_loss':>10s} {'d_loss':>10s}", flush=True)
    for r in summary:
        gl = f"{r['g_loss']:.4f}" if r["g_loss"] is not None else "—"
        dl = f"{r['d_loss']:.4f}" if r["d_loss"] is not None else "—"
        print(f"[gan] {r['arch']:8s} {r['params'] / 1e6:9.2f}M {gl:>10s} {dl:>10s}", flush=True)
