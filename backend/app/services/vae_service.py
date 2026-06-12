"""Servicio del VAE: estado, entrenamiento (generador de eventos), checkpoint, reconstrucción
y muestreo del prior. Espejo de `ae_service.py` (mismo patrón de hilo+cola para SSE), con las
diferencias propias del VAE: pérdida = reconstrucción + β·KL, latente probabilístico (μ, logσ²)
y generación por muestreo del prior N(0,I) (CLAUDE.md §3.1, §3.2, §3.3, §4.2)."""

from __future__ import annotations

import queue
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn

from .. import imaging, metrics as metrics_mod
from ..data import dataset
from ..device import get_device
from ..models.vae import VAE_ARCHS, build_vae, count_params
from ..seeding import set_seed

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
# Checkpoint demo "legacy" (sin variantes): se trata como el demo de "basico".
LEGACY_DEMO_PATH = CKPT_DIR / "vae_demo.pt"


def demo_path(arch: str) -> Path:
    """Ruta del checkpoint demo por variante, p. ej. vae_basico_demo.pt."""
    return CKPT_DIR / f"vae_{arch}_demo.pt"


QUICK_N = 6000          # subconjunto para el modo "quick" (solo cambia el tamaño de datos)
DEMO_EPOCHS = 12        # checkpoint demo (modo full)
LOG_EVERY = 25          # steps entre eventos de progreso
EARLY_STOP_MIN_DELTA = 1e-4  # mejora mínima de pérdida por epoch para resetear la paciencia
PREVIEW_IDS = [12, 800, 4096, 20000]  # caras fijas para la vista previa
EMBED_SAMPLE = 1200     # nº de z muestreados para el mapa latente / vecinos / clusters
METRICS_N = 256         # tamaño del held-out fijo para PSNR/SSIM/MSE
METRICS_SEED = 1234     # semilla fija del held-out de métricas (independiente del entrenamiento)

# --- escala de la pérdida (mejora v3, ver «Lista de Mejoras.md» §3) -------------------
# La reconstrucción (MSELoss/L1Loss) se PROMEDIA sobre los 12.288 valores de la imagen,
# pero la KL se SUMA sobre las dimensiones del latente. Sin reescalar, con β=1 la KL pesa
# miles de veces más que en el ELBO real → colapso del posterior (z deja de informar y el
# decoder pinta siempre la "cara media"). v3: la KL se divide por el nº de píxeles, de modo
# que β=1 ≈ ELBO bien equilibrado y el slider de β recupera su significado de β-VAE.
PIXELS = 3 * 64 * 64    # 12.288 valores por imagen
KL_WARMUP_FRAC = 0.3    # fracción inicial de epochs en la que β sube linealmente 0→β
                        # (deja que el decoder aprenda a reconstruir antes de regularizar)


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) ‖ N(0,I)) en nats por muestra (suma sobre dims, media sobre el lote).

    KL = -½ · mean( sum_j (1 + logσ²_j − μ_j² − exp(logσ²_j)) ).
    Mide cuánto se aleja el posterior del prior gaussiano estándar. Para usarla en la
    pérdida se divide por PIXELS (misma escala que el MSE medio por píxel).
    """
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


@dataclass
class VAEHyperParams:
    latent_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    loss: str = "mse"  # "mse" | "l1"  (término de reconstrucción)
    beta: float = 1.0  # peso del término KL (β-VAE)
    batch_size: int = 256
    arch: str = "basico"  # "basico" | "grande"

    def sanitized(self) -> "VAEHyperParams":
        return VAEHyperParams(
            latent_dim=int(max(8, min(512, self.latent_dim))),
            learning_rate=float(max(1e-5, min(1e-1, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            loss="l1" if str(self.loss).lower() == "l1" else "mse",
            beta=float(max(0.0, min(10.0, self.beta))),
            batch_size=int(max(16, min(512, self.batch_size))),
            arch=self.arch if self.arch in VAE_ARCHS else "basico",
        )


class VAEService:
    """Mantiene el modelo VAE y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.model: nn.Module | None = None
        self.hp = VAEHyperParams()
        self.seed = 42
        self.trained = False
        self.outdated = False
        self.training = False
        self.loss_history: list[dict] = []
        self._cancel = False
        # embedding cacheado (muestra de μ) para mapa latente / vecinos / clusters
        self._embed_ids: np.ndarray | None = None
        self._embed_Z: np.ndarray | None = None
        self._cluster_labels: np.ndarray | None = None

    # ---------- estado ----------
    def status(self) -> dict:
        return {
            "model": "vae",
            "trained": self.trained,
            "outdated": self.outdated,
            "training": self.training,
            "seed": self.seed,
            "device": self.device.type,
            "arch": self.hp.arch,
            "archs": list(VAE_ARCHS),
            "hyperparams": asdict(self.hp),
            "num_params": count_params(self.model) if self.model is not None else 0,
            "loss_history": self.loss_history,
        }

    def set_hyperparams(self, partial: dict) -> dict:
        """Actualiza hiperparámetros y marca el modelo como 'requiere reentrenar' si cambian."""
        new = replace(self.hp, **{k: v for k, v in partial.items() if hasattr(self.hp, k)}).sanitized()
        if self.trained and asdict(new) != asdict(self.hp):
            self.outdated = True
        self.hp = new
        return self.status()

    def cancel(self) -> None:
        self._cancel = True

    # ---------- datos ----------
    def _iter_batches(self, indices: np.ndarray, batch_size: int) -> Iterator[torch.Tensor]:
        arr = dataset.load_array()
        for i in range(0, len(indices), batch_size):
            idx = indices[i : i + batch_size]
            chunk = np.asarray(arr[idx])  # (B,64,64,3) uint8
            t = torch.from_numpy(chunk).float().div_(255).permute(0, 3, 1, 2).contiguous()
            yield t.to(self.device)

    def _preview(self, model: nn.Module) -> list[str]:
        """Reconstrucción determinista (vía μ) de unas caras fijas para la vista previa."""
        was_training = model.training
        model.eval()
        arr = dataset.load_array()
        n = dataset.count()
        ids = [i for i in PREVIEW_IDS if i < n][:4]
        with torch.no_grad():
            xs = torch.stack([imaging.uint8_to_tensor(np.asarray(arr[i])) for i in ids]).to(self.device)
            xh = model.decode(model.encode_mu(xs))
        if was_training:
            model.train()
        return [imaging.tensor_to_b64(xh[k]) for k in range(len(ids))]

    # ---------- entrenamiento ----------
    def iter_train(
        self, mode: str, hp: VAEHyperParams, seed: int, early_stop: bool = True, patience: int = 5
    ) -> Iterator[dict]:
        """Entrena emitiendo eventos {type, epoch, step, loss, recon_loss, kl_loss, preview?}.

        La pérdida es `recon + β·KL`. En cada epoch se emiten también `recon_loss` y `kl_loss`
        por separado para poder graficar el trade-off reconstrucción ↔ regularidad.

        El número de epochs del slider se respeta siempre; "quick" solo reduce el tamaño del
        subconjunto. Con early_stop, para antes si la pérdida TOTAL no mejora durante `patience`
        epochs.
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
            epochs = hp.epochs

            model = build_vae(hp.latent_dim, hp.arch).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
            recon_crit: nn.Module = nn.L1Loss() if hp.loss == "l1" else nn.MSELoss()

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

            # Warmup de β (v3): durante el primer KL_WARMUP_FRAC de epochs, β sube linealmente
            # de 0 a su valor. Así el decoder aprende primero a reconstruir y la regularización
            # entra después (evita que la KL gane la carrera al principio y colapse el posterior).
            warmup_epochs = max(1.0, round(KL_WARMUP_FRAC * epochs))
            for epoch in range(1, epochs + 1):
                beta_t = hp.beta * min(1.0, epoch / warmup_epochs)
                rng.shuffle(indices)
                run_total = torch.zeros((), device=self.device)
                run_recon = torch.zeros((), device=self.device)
                run_kl = torch.zeros((), device=self.device)
                nb = 0
                for batch in self._iter_batches(indices, hp.batch_size):
                    opt.zero_grad(set_to_none=True)
                    recon, mu, logvar, _ = model(batch)
                    recon_loss = recon_crit(recon, batch)
                    # KL por píxel (v3): misma escala que el MSE medio → β=1 ≈ ELBO equilibrado.
                    kl_loss = _kl_divergence(mu, logvar) / PIXELS
                    loss = recon_loss + beta_t * kl_loss
                    loss.backward()
                    opt.step()
                    run_total += loss.detach()
                    run_recon += recon_loss.detach()
                    run_kl += kl_loss.detach()
                    nb += 1
                    global_step += 1
                    if global_step % LOG_EVERY == 0:
                        yield {"type": "step", "epoch": epoch, "step": global_step,
                               "loss": float((run_total / nb).item()),
                               "recon_loss": float((run_recon / nb).item()),
                               "kl_loss": float((run_kl / nb).item())}
                    if self._cancel:
                        yield {"type": "cancelled", "epoch": epoch, "step": global_step}
                        return
                denom = max(nb, 1)
                epoch_loss = float((run_total / denom).item())
                epoch_recon = float((run_recon / denom).item())
                epoch_kl = float((run_kl / denom).item())
                history.append({"epoch": epoch, "loss": epoch_loss,
                                "recon": epoch_recon, "kl": epoch_kl})
                yield {"type": "epoch", "epoch": epoch, "epochs": epochs, "step": global_step,
                       "loss": epoch_loss, "recon_loss": epoch_recon, "kl_loss": epoch_kl,
                       "beta_t": round(beta_t, 4), "preview": self._preview(model)}
                if early_stop:
                    if epoch_loss < best - EARLY_STOP_MIN_DELTA:
                        best = epoch_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            stopped_early = True
                            break

            # finalizar: adoptar el modelo entrenado (en memoria) — NO se escribe el checkpoint
            model.eval()
            self.model = model
            self.hp = hp
            self.seed = seed
            self.trained = True
            self.outdated = False
            self.loss_history = history
            self._invalidate_embedding()
            # IMPORTANTE: como en el AE, el entrenamiento en la app NO toca el checkpoint demo. El
            # modelo entrenado vive en memoria durante esta sesión del servidor; al reiniciar se
            # vuelve al demo pristino. Así, experimentar nunca degrada el estado base.
            yield {"type": "done", "epochs": epochs, "epochs_run": len(history),
                   "stopped_early": stopped_early,
                   "loss": history[-1]["loss"] if history else None,
                   "loss_history": history}
        finally:
            self.training = False

    def start_training(
        self, mode: str, hp: VAEHyperParams, seed: int, early_stop: bool = True
    ) -> "queue.Queue":
        """Arranca el entrenamiento en un hilo propio y devuelve una cola de eventos.

        Desacopla el ciclo de vida del entrenamiento de la conexión del cliente: aunque el
        navegador se desconecte, el hilo termina (o se cancela) y libera el estado. El None
        final es el centinela de fin de stream.
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

        threading.Thread(target=worker, name="vae-train", daemon=True).start()
        return q

    # ---------- búsqueda en rejilla (grid search) ----------
    def _eval_mse(self, model: nn.Module, ids: np.ndarray) -> float:
        """MSE de reconstrucción (determinista, vía μ) por píxel sobre un conjunto de evaluación."""
        model.eval()
        tot = 0.0
        cnt = 0
        with torch.no_grad():
            for batch in self._iter_batches(ids, 256):
                recon = model.decode(model.encode_mu(batch))
                tot += torch.mean((recon - batch) ** 2, dim=[1, 2, 3]).sum().item()
                cnt += int(batch.shape[0])
        return tot / max(cnt, 1)

    def iter_gridsearch(self, scope: str, epochs: int, grid: dict, seed: int) -> Iterator[dict]:
        """Entrena cada combinación de la rejilla (con el β configurado) y la evalúa por MSE de
        reconstrucción en un held-out fijo. NO adopta ningún modelo ni toca el checkpoint: solo
        busca la mejor configuración."""
        if self.training:
            yield {"type": "error", "message": "Ya hay una operación en curso"}
            return
        self.training = True
        self._cancel = False
        try:
            epochs = int(max(1, min(15, epochs)))
            beta = float(max(0.0, min(10.0, self.hp.beta)))
            lds = sorted({int(max(8, min(512, x))) for x in grid.get("latent_dim", [128])})
            lrs = sorted({float(max(1e-5, min(1e-1, x))) for x in grid.get("learning_rate", [1e-3])})
            losses = [x for x in ["mse", "l1"] if x in set(grid.get("loss", ["mse"]))] or ["mse"]
            combos = [(ld, lr, ls) for ld in lds for lr in lrs for ls in losses][:16]
            total = len(combos)
            arch = self.hp.arch  # la rejilla varía ld/lr/loss sobre la variante actual

            n = dataset.count()
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
            eval_ids = perm[:1000]
            pool = perm[1000:]
            train_ids = pool if scope == "full" else pool[: min(QUICK_N, len(pool))]

            yield {"type": "start", "total": total, "scope": scope, "epochs": epochs,
                   "eval_n": int(len(eval_ids)), "train_n": int(len(train_ids))}

            results: list[dict] = []
            best: dict | None = None
            for idx, (ld, lr, ls) in enumerate(combos):
                if self._cancel:
                    yield {"type": "cancelled"}
                    return
                set_seed(seed)
                model = build_vae(ld, arch).to(self.device)
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                recon_crit: nn.Module = nn.L1Loss() if ls == "l1" else nn.MSELoss()
                ids = train_ids.copy()
                for ep in range(epochs):
                    np.random.default_rng(seed + ep + 1).shuffle(ids)
                    for batch in self._iter_batches(ids, 256):
                        opt.zero_grad(set_to_none=True)
                        recon, mu, logvar, _ = model(batch)
                        # misma escala v3 que en iter_train: KL por píxel
                        loss = recon_crit(recon, batch) + beta * _kl_divergence(mu, logvar) / PIXELS
                        loss.backward()
                        opt.step()
                        if self._cancel:
                            yield {"type": "cancelled"}
                            return
                    yield {"type": "progress", "index": idx, "total": total,
                           "epoch": ep + 1, "epochs": epochs}
                cfg = {"latent_dim": ld, "learning_rate": lr, "loss": ls, "epochs": epochs, "arch": arch}
                res = {**cfg, "eval_mse": self._eval_mse(model, eval_ids)}
                results.append(res)
                if best is None or res["eval_mse"] < best["eval_mse"]:
                    best = res
                yield {"type": "trial", "index": idx, "total": total, "result": res, "best": best}

            results.sort(key=lambda r: r["eval_mse"])
            yield {"type": "done", "results": results, "best": results[0] if results else None}
        finally:
            self.training = False

    def start_gridsearch(self, scope: str, epochs: int, grid: dict, seed: int) -> "queue.Queue":
        q: "queue.Queue" = queue.Queue(maxsize=256)
        if self.training:
            q.put({"type": "error", "message": "Ya hay una operación en curso"})
            q.put(None)
            return q

        def worker() -> None:
            try:
                for ev in self.iter_gridsearch(scope, epochs, grid, seed):
                    q.put(ev)
            except Exception as exc:  # pragma: no cover - defensivo
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, name="vae-gridsearch", daemon=True).start()
        return q

    # ---------- checkpoint ----------
    def save_checkpoint(self, path: Path | None = None) -> None:
        """Guarda el modelo en `vae_<arch>_demo.pt`. Solo lo usa el `__main__` generador del demo."""
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

    def _load_path(self, path: Path) -> bool:
        """Carga un checkpoint desde una ruta concreta. Tolera arch desconocida (devuelve False)."""
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device)
        hp = VAEHyperParams(**ckpt["hyperparams"])
        # arch puede venir suelta (v2) o dentro de hyperparams; v1 (legacy) → "basico".
        arch = ckpt.get("arch", getattr(hp, "arch", "basico"))
        if arch not in VAE_ARCHS:
            return False  # checkpoint de una variante que ya no existe: no crashear
        hp = replace(hp, arch=arch).sanitized()
        try:
            model = build_vae(hp.latent_dim, hp.arch).to(self.device)
            model.load_state_dict(ckpt["state_dict"])
        except (RuntimeError, KeyError):
            return False  # incompatibilidad de pesos: degradar sin romper
        model.eval()
        self.model = model
        self.hp = hp
        self.seed = ckpt.get("seed", 42)
        self.loss_history = ckpt.get("loss_history", [])
        self.trained = True
        self.outdated = False
        self._invalidate_embedding()
        return True

    def load_demo(self, arch: str) -> bool:
        """Carga el checkpoint demo de la variante `arch` (`vae_<arch>_demo.pt`).

        Devuelve True si se cargó. Compatibilidad: si se pide "basico" y solo existe el viejo
        `vae_demo.pt`, se usa ese. Lo usa la UI para cambiar de variante al instante.
        """
        if arch not in VAE_ARCHS:
            return False
        path = demo_path(arch)
        if not path.exists() and arch == "basico" and LEGACY_DEMO_PATH.exists():
            path = LEGACY_DEMO_PATH
        return self._load_path(path)

    def load_checkpoint(self, arch: str | None = None) -> bool:
        """Carga el demo de la variante `arch` (o la actual). Compatible con el viejo `vae_demo.pt`.

        Al arrancar el servidor se llama sin argumentos: intenta la variante actual y, si no hay,
        cae a "basico" (que cubre el legacy `vae_demo.pt`) para que la app siempre tenga un demo.
        """
        if arch is not None:
            return self.load_demo(arch)
        if self.load_demo(self.hp.arch):
            return True
        if self.hp.arch != "basico":
            return self.load_demo("basico")
        return False

    # ---------- reconstrucción ----------
    def reconstruct(self, ids: list[int], noise: float = 0.0) -> list[dict]:
        """Reconstruye usando μ (determinista). Si noise>0, suma ruido a μ antes de decodificar."""
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        arr = dataset.load_array()
        n = dataset.count()
        out: list[dict] = []
        self.model.eval()
        with torch.no_grad():
            for i in ids:
                if i < 0 or i >= n:
                    continue
                x = imaging.uint8_to_tensor(np.asarray(arr[i])).to(self.device)
                z = self.model.encode_mu(x.unsqueeze(0))
                if noise > 0:
                    z = z + noise * torch.randn_like(z)
                recon = self.model.decode(z)[0]
                out.append(
                    {
                        "id": int(i),
                        "original": imaging.tensor_to_b64(x),
                        "reconstruction": imaging.tensor_to_b64(recon),
                        "diff": imaging.diff_b64(x.cpu(), recon.cpu()),
                    }
                )
        return out

    def reconstruct_tensor(self, x: torch.Tensor) -> dict:
        """Reconstruye un tensor CHW [0,1] (p. ej. una imagen subida), vía μ."""
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        self.model.eval()
        with torch.no_grad():
            recon = self.model.decode(self.model.encode_mu(x.unsqueeze(0).to(self.device)))[0]
        return {
            "original": imaging.tensor_to_b64(x),
            "reconstruction": imaging.tensor_to_b64(recon),
            "diff": imaging.diff_b64(x.cpu(), recon.cpu()),
        }

    # ---------- métricas de reconstrucción ----------
    def _heldout_ids(self, n: int) -> np.ndarray:
        """Subconjunto held-out FIJO por semilla (independiente del entrenamiento)."""
        total = dataset.count()
        rng = np.random.default_rng(METRICS_SEED)
        n = int(max(1, min(n, total)))
        return np.sort(rng.choice(total, size=n, replace=False))

    def metrics(self, n: int = METRICS_N, n_examples: int = 6) -> dict:
        """Reconstruye un held-out fijo y devuelve MSE/PSNR/SSIM agregados + unos ejemplos.

        Usa la reconstrucción determinista (vía μ, sin muestrear): mide la calidad real del
        modelo sin el ruido del término estocástico.
        """
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        arr = dataset.load_array()
        ids = self._heldout_ids(n)
        self.model.eval()

        mse_tot = psnr_tot = ssim_tot = 0.0
        cnt = 0
        examples: list[dict] = []
        ex_ids = set(ids[:n_examples].tolist())
        with torch.no_grad():
            for i in range(0, len(ids), 128):
                idx = ids[i : i + 128]
                x = (
                    torch.from_numpy(np.asarray(arr[idx])).float().div(255)
                    .permute(0, 3, 1, 2).contiguous().to(self.device)
                )
                xhat = self.model.decode(self.model.encode_mu(x)).clamp(0, 1)
                b = x.shape[0]
                mse_tot += metrics_mod.mse(x, xhat) * b
                psnr_tot += metrics_mod.psnr(x, xhat) * b
                ssim_tot += metrics_mod.ssim(x, xhat) * b
                cnt += b
                for k, pid in enumerate(idx.tolist()):
                    if pid in ex_ids:
                        xk, xhk = x[k], xhat[k]
                        examples.append(
                            {
                                "id": int(pid),
                                "original": imaging.tensor_to_b64(xk),
                                "reconstruction": imaging.tensor_to_b64(xhk),
                                "diff": imaging.diff_b64(xk.cpu(), xhk.cpu()),
                                "psnr": round(metrics_mod.psnr(xk, xhk), 2),
                                "ssim": round(metrics_mod.ssim(xk, xhk), 4),
                            }
                        )
        examples.sort(key=lambda e: e["id"])
        return {
            "mse": mse_tot / max(cnt, 1),
            "psnr": psnr_tot / max(cnt, 1),
            "ssim": ssim_tot / max(cnt, 1),
            "n": int(cnt),
            "arch": self.hp.arch,
            "latent_dim": self.hp.latent_dim,
            "examples": examples,
        }

    # ---------- generación (muestreo del prior) ----------
    def generate(self, n: int, seed: int | None = None) -> list[str]:
        """Muestrea n imágenes del prior z~N(0,I) y las devuelve como PNG base64.

        Es el rasgo clave del VAE frente al AE: como el espacio latente está regularizado
        hacia N(0,I), podemos muestrear del prior y obtener caras nuevas plausibles.
        """
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        n = int(max(1, min(64, n)))
        if seed is not None:
            set_seed(int(seed))
        imgs = self.model.sample(n, self.device)
        return [imaging.tensor_to_b64(imgs[k]) for k in range(n)]

    # ---------- latente: embedding, proyección, vecinos, clusters ----------
    def _invalidate_embedding(self) -> None:
        self._embed_ids = None
        self._embed_Z = None
        self._cluster_labels = None

    def ensure_embedding(self, sample_size: int = EMBED_SAMPLE) -> None:
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        if self._embed_Z is not None:
            return
        n = dataset.count()
        rng = np.random.default_rng(self.seed)
        ids = np.sort(rng.choice(n, size=min(sample_size, n), replace=False))
        arr = dataset.load_array()
        self.model.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(ids), 256):
                idx = ids[i : i + 256]
                t = torch.from_numpy(np.asarray(arr[idx])).float().div(255)
                t = t.permute(0, 3, 1, 2).to(self.device)
                chunks.append(self.model.encode_mu(t).cpu().numpy())
        self._embed_ids = ids
        self._embed_Z = np.concatenate(chunks, 0).astype(np.float32)
        self._cluster_labels = None

    def _points(self, method: str) -> dict:
        from . import latent

        assert self._embed_Z is not None and self._embed_ids is not None
        xy, used = latent.project_2d(self._embed_Z, method=method, seed=self.seed)
        znorm = np.linalg.norm(self._embed_Z, axis=1)
        labels = self._cluster_labels
        points = [
            {
                "id": int(pid),
                "x": float(xy[k, 0]),
                "y": float(xy[k, 1]),
                "znorm": float(znorm[k]),
                "cluster": int(labels[k]) if labels is not None else None,
            }
            for k, pid in enumerate(self._embed_ids)
        ]
        return {
            "points": points,
            "method": used,
            "n_clusters": int(labels.max() + 1) if labels is not None else 0,
        }

    def projection(self, method: str = "pca") -> dict:
        self.ensure_embedding()
        return self._points(method)

    def cluster(self, n_clusters: int, method: str = "pca") -> dict:
        from . import latent

        self.ensure_embedding()
        assert self._embed_Z is not None
        self._cluster_labels = latent.kmeans_labels(self._embed_Z, n_clusters, seed=self.seed)
        return self._points(method)

    def neighbors(self, query_id: int, k: int) -> dict:
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        self.ensure_embedding()
        assert self._embed_Z is not None and self._embed_ids is not None
        arr = dataset.load_array()
        n = dataset.count()
        query_id = int(max(0, min(n - 1, query_id)))
        k = int(max(1, min(24, k)))
        self.model.eval()
        with torch.no_grad():
            xq = imaging.uint8_to_tensor(np.asarray(arr[query_id])).unsqueeze(0).to(self.device)
            zq = self.model.encode_mu(xq).cpu().numpy()[0]
        d = np.linalg.norm(self._embed_Z - zq[None, :], axis=1)
        neigh: list[dict] = []
        for j in np.argsort(d):
            pid = int(self._embed_ids[j])
            if pid == query_id:
                continue
            neigh.append(
                {"id": pid, "distance": float(d[j]), "image": imaging.encode_png_b64(np.asarray(arr[pid]))}
            )
            if len(neigh) >= k:
                break
        return {
            "query": {"id": query_id, "image": imaging.encode_png_b64(np.asarray(arr[query_id]))},
            "neighbors": neigh,
        }

    def interpolate(self, a_id: int, b_id: int, steps: int) -> dict:
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        arr = dataset.load_array()
        n = dataset.count()
        a_id = int(max(0, min(n - 1, a_id)))
        b_id = int(max(0, min(n - 1, b_id)))
        steps = int(max(2, min(12, steps)))
        self.model.eval()
        with torch.no_grad():
            za = self.model.encode_mu(imaging.uint8_to_tensor(np.asarray(arr[a_id])).unsqueeze(0).to(self.device))
            zb = self.model.encode_mu(imaging.uint8_to_tensor(np.asarray(arr[b_id])).unsqueeze(0).to(self.device))
            frames = []
            for s in range(steps):
                alpha = s / (steps - 1)
                img = self.model.decode((1 - alpha) * za + alpha * zb)[0]
                frames.append({"alpha": round(alpha, 3), "image": imaging.tensor_to_b64(img)})
        return {"a_id": a_id, "b_id": b_id, "steps": steps, "frames": frames}


# Singleton compartido por los routers.
vae_service = VAEService()


if __name__ == "__main__":
    # Genera el checkpoint demo de LAS DOS variantes:  python -m app.services.vae_service
    # Cada una entrena en modo full acotado (DEMO_EPOCHS, β=1.0) y se guarda en vae_<arch>_demo.pt.
    summary: list[dict] = []
    for arch in VAE_ARCHS:
        svc = VAEService()
        print(f"[vae] === variante '{arch}' · full {DEMO_EPOCHS} epochs (β=1.0) en {svc.device.type} ===",
              flush=True)
        hp = VAEHyperParams(epochs=DEMO_EPOCHS, beta=1.0, batch_size=256, arch=arch)
        for ev in svc.iter_train("full", hp, seed=42):
            if ev["type"] == "epoch":
                print(
                    f"[vae]   [{arch}] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}"
                    f" (recon {ev['recon_loss']:.5f} · kl {ev['kl_loss']:.5f})",
                    flush=True,
                )
            elif ev["type"] == "done":
                print(f"[vae]   [{arch}] listo · loss final {ev['loss']:.5f}", flush=True)
        path = demo_path(arch)
        svc.save_checkpoint(path)
        print(f"[vae]   [{arch}] checkpoint guardado en {path}", flush=True)
        m = svc.metrics()
        print(f"[vae]   [{arch}] held-out (n={m['n']}): "
              f"MSE {m['mse']:.5f} · PSNR {m['psnr']:.2f} dB · SSIM {m['ssim']:.4f} "
              f"· params {count_params(svc.model) / 1e6:.2f}M", flush=True)
        summary.append({"arch": arch, **{k: m[k] for k in ("mse", "psnr", "ssim")},
                        "params": count_params(svc.model)})

    print("\n[vae] === Resumen de variantes (held-out fijo) ===", flush=True)
    print(f"[vae] {'arch':8s} {'params':>10s} {'MSE':>10s} {'PSNR(dB)':>10s} {'SSIM':>8s}", flush=True)
    for r in summary:
        print(f"[vae] {r['arch']:8s} {r['params'] / 1e6:9.2f}M {r['mse']:10.5f} "
              f"{r['psnr']:10.2f} {r['ssim']:8.4f}", flush=True)
