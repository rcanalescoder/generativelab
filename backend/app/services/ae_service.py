"""Servicio del Autoencoder: estado, entrenamiento (generador de eventos), checkpoint y
reconstrucción. El generador de entrenamiento se reutiliza tanto para el endpoint SSE como
para generar el checkpoint demo (CLAUDE.md §3.1, §3.2, §3.3)."""

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
from ..device import device_guard, get_device
from ..models.ae import ARCHS, build_autoencoder, count_params
from ..seeding import set_seed

CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"
# Checkpoint demo "legacy" (sin variantes): se trata como el demo de "basico".
LEGACY_DEMO_PATH = CKPT_DIR / "ae_demo.pt"


def demo_path(arch: str) -> Path:
    """Ruta del checkpoint demo por variante, p. ej. ae_basico_demo.pt."""
    return CKPT_DIR / f"ae_{arch}_demo.pt"

QUICK_N = 6000          # subconjunto para el modo "quick" (solo cambia el tamaño de datos)
DEMO_EPOCHS = 10        # checkpoint demo (modo full)
LOG_EVERY = 25          # steps entre eventos de progreso
EARLY_STOP_MIN_DELTA = 1e-4  # mejora mínima de pérdida por epoch para resetear la paciencia
PREVIEW_IDS = [12, 800, 4096, 20000]  # caras fijas para la vista previa
EMBED_SAMPLE = 1200     # nº de z muestreados para el mapa latente / vecinos / clusters
METRICS_N = 256         # tamaño del held-out fijo para PSNR/SSIM/MSE
METRICS_SEED = 1234     # semilla fija del held-out de métricas (independiente del entrenamiento)


@dataclass
class AEHyperParams:
    latent_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    loss: str = "mse"  # "mse" | "l1"
    batch_size: int = 256
    arch: str = "basico"  # "basico" | "grande" | "unet"

    def sanitized(self) -> "AEHyperParams":
        return AEHyperParams(
            latent_dim=int(max(8, min(512, self.latent_dim))),
            learning_rate=float(max(1e-5, min(1e-1, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            loss="l1" if str(self.loss).lower() == "l1" else "mse",
            batch_size=int(max(16, min(512, self.batch_size))),
            arch=self.arch if self.arch in ARCHS else "basico",
        )


class AEService:
    """Mantiene el modelo AE y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.model: nn.Module | None = None
        self.hp = AEHyperParams()
        self.seed = 42
        self.trained = False
        self.outdated = False
        self.training = False
        self.loss_history: list[dict] = []
        self._cancel = False
        # embedding cacheado (muestra de z) para mapa latente / vecinos / clusters
        self._embed_ids: np.ndarray | None = None
        self._embed_Z: np.ndarray | None = None
        self._cluster_labels: np.ndarray | None = None

    # ---------- estado ----------
    def status(self) -> dict:
        return {
            "model": "autoencoder",
            "trained": self.trained,
            "outdated": self.outdated,
            "training": self.training,
            "seed": self.seed,
            "device": self.device.type,
            "arch": self.hp.arch,
            "archs": list(ARCHS),
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
            with device_guard():  # el traslado a la GPU también es Metal: serialízalo
                t = t.to(self.device)
            yield t

    def _preview(self, model: nn.Module) -> list[str]:
        was_training = model.training
        model.eval()
        arr = dataset.load_array()
        n = dataset.count()
        ids = [i for i in PREVIEW_IDS if i < n][:4]
        with device_guard(), torch.no_grad():
            xs = torch.stack([imaging.uint8_to_tensor(np.asarray(arr[i])) for i in ids]).to(self.device)
            xh, _ = model(xs)
        if was_training:
            model.train()
        return [imaging.tensor_to_b64(xh[k]) for k in range(len(ids))]

    # ---------- entrenamiento ----------
    def iter_train(
        self, mode: str, hp: AEHyperParams, seed: int, early_stop: bool = True, patience: int = 5
    ) -> Iterator[dict]:
        """Entrena emitiendo eventos {type, epoch, step, loss, preview?}.

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
            n_total = dataset.count()
            quick = mode == "quick"
            n_use = min(QUICK_N, n_total) if quick else n_total
            epochs = hp.epochs

            model = build_autoencoder(hp.latent_dim, hp.arch).to(self.device)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
            crit: nn.Module = nn.L1Loss() if hp.loss == "l1" else nn.MSELoss()

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
                for batch in self._iter_batches(indices, hp.batch_size):
                    # Serializa el paso frente a la inferencia (MPS no es thread-safe; ver device.py).
                    with device_guard():
                        opt.zero_grad(set_to_none=True)
                        recon, _ = model(batch)
                        loss = crit(recon, batch)
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
                       "loss": epoch_loss, "preview": self._preview(model)}
                if early_stop:
                    if epoch_loss < best - EARLY_STOP_MIN_DELTA:
                        best = epoch_loss
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= patience:
                            stopped_early = True
                            break

            # finalizar: adoptar el modelo entrenado y guardar
            model.eval()
            self.model = model
            self.hp = hp
            self.seed = seed
            self.trained = True
            self.outdated = False
            self.loss_history = history
            self._invalidate_embedding()
            # IMPORTANTE: el entrenamiento en la app NO toca el checkpoint demo. El modelo
            # entrenado vive en memoria durante esta sesión del servidor; al reiniciar se
            # vuelve al demo pristino. Así, experimentar nunca degrada el estado base.
            yield {"type": "done", "epochs": epochs, "epochs_run": len(history),
                   "stopped_early": stopped_early,
                   "loss": history[-1]["loss"] if history else None,
                   "loss_history": history}
        finally:
            self.training = False

    def start_training(
        self, mode: str, hp: AEHyperParams, seed: int, early_stop: bool = True
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

        threading.Thread(target=worker, name="ae-train", daemon=True).start()
        return q

    # ---------- búsqueda en rejilla (grid search) ----------
    def _eval_mse(self, model: nn.Module, ids: np.ndarray) -> float:
        """MSE de reconstrucción por píxel sobre un conjunto de evaluación."""
        model.eval()
        tot = 0.0
        cnt = 0
        with device_guard(), torch.no_grad():
            for batch in self._iter_batches(ids, 256):
                recon, _ = model(batch)
                tot += torch.mean((recon - batch) ** 2, dim=[1, 2, 3]).sum().item()
                cnt += int(batch.shape[0])
        return tot / max(cnt, 1)

    def iter_gridsearch(self, scope: str, epochs: int, grid: dict, seed: int) -> Iterator[dict]:
        """Entrena cada combinación de la rejilla y la evalúa por MSE en un held-out fijo.
        NO adopta ningún modelo ni toca el checkpoint: solo busca la mejor configuración."""
        if self.training:
            yield {"type": "error", "message": "Ya hay una operación en curso"}
            return
        self.training = True
        self._cancel = False
        try:
            epochs = int(max(1, min(15, epochs)))
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
                model = build_autoencoder(ld, arch).to(self.device)
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                crit: nn.Module = nn.L1Loss() if ls == "l1" else nn.MSELoss()
                ids = train_ids.copy()
                for ep in range(epochs):
                    np.random.default_rng(seed + ep + 1).shuffle(ids)
                    for batch in self._iter_batches(ids, 256):
                        with device_guard():
                            opt.zero_grad(set_to_none=True)
                            recon, _ = model(batch)
                            loss = crit(recon, batch)
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

        threading.Thread(target=worker, name="ae-gridsearch", daemon=True).start()
        return q

    # ---------- checkpoint ----------
    def save_checkpoint(self, path: Path | None = None) -> None:
        """Guarda el modelo en `ae_<arch>_demo.pt`. Solo lo usa el `__main__` generador del demo."""
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
        hp = AEHyperParams(**ckpt["hyperparams"])
        # arch puede venir suelta (v2) o dentro de hyperparams; v1 (legacy) → "basico".
        arch = ckpt.get("arch", getattr(hp, "arch", "basico"))
        if arch not in ARCHS:
            return False  # checkpoint de una variante que ya no existe: no crashear
        hp = replace(hp, arch=arch).sanitized()
        try:
            model = build_autoencoder(hp.latent_dim, hp.arch).to(self.device)
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
        """Carga el checkpoint demo de la variante `arch` (`ae_<arch>_demo.pt`).

        Devuelve True si se cargó. Compatibilidad: si se pide "basico" y solo existe el viejo
        `ae_demo.pt`, se usa ese. Lo usa la UI para cambiar de variante al instante.
        """
        if arch not in ARCHS:
            return False
        path = demo_path(arch)
        if not path.exists() and arch == "basico" and LEGACY_DEMO_PATH.exists():
            path = LEGACY_DEMO_PATH
        return self._load_path(path)

    def load_checkpoint(self, arch: str | None = None) -> bool:
        """Carga el demo de la variante `arch` (o la actual). Compatible con el viejo `ae_demo.pt`.

        Al arrancar el servidor se llama sin argumentos: intenta la variante actual y, si no hay,
        cae a "basico" (que cubre el legacy `ae_demo.pt`) para que la app siempre tenga un demo.
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
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        arr = dataset.load_array()
        n = dataset.count()
        out: list[dict] = []
        self.model.eval()
        with device_guard(), torch.no_grad():
            for i in ids:
                if i < 0 or i >= n:
                    continue
                x = imaging.uint8_to_tensor(np.asarray(arr[i])).to(self.device)
                if noise > 0:
                    # Camino z+ruido: añade ruido al cuello z y reconstruye.
                    # En la U-Net decodificamos con los skips REALES de la imagen: así la cara
                    # sigue siendo coherente (no un blob gris) y se aprecia que perturbar el
                    # cuello apenas la cambia, porque los skips llevan la información. En
                    # básico/grande el cuello es todo el código, así que el ruido sí degrada.
                    if hasattr(self.model, "encode_with_skips"):
                        z, skips = self.model.encode_with_skips(x.unsqueeze(0))
                        z = z + noise * torch.randn_like(z)
                        recon = self.model.decode(z, skips)[0]
                    else:
                        z = self.model.encode(x.unsqueeze(0))
                        z = z + noise * torch.randn_like(z)
                        recon = self.model.decode(z)[0]
                else:
                    # Reconstrucción nítida: forward usa skips si la variante (U-Net) los tiene.
                    recon = self.model(x.unsqueeze(0))[0][0]
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
        """Reconstruye un tensor CHW [0,1] (p. ej. una imagen subida)."""
        if self.model is None:
            raise RuntimeError("modelo no entrenado")
        self.model.eval()
        with device_guard(), torch.no_grad():
            recon = self.model(x.unsqueeze(0).to(self.device))[0][0]
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

        Usa la reconstrucción nítida (forward, con skips en U-Net): mide la calidad real del
        modelo, no el camino degradado a través del cuello.
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
        with device_guard(), torch.no_grad():
            for i in range(0, len(ids), 128):
                idx = ids[i : i + 128]
                x = (
                    torch.from_numpy(np.asarray(arr[idx])).float().div(255)
                    .permute(0, 3, 1, 2).contiguous().to(self.device)
                )
                xhat = self.model(x)[0].clamp(0, 1)
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
        with device_guard(), torch.no_grad():
            for i in range(0, len(ids), 256):
                idx = ids[i : i + 256]
                t = torch.from_numpy(np.asarray(arr[idx])).float().div(255)
                t = t.permute(0, 3, 1, 2).to(self.device)
                chunks.append(self.model.encode(t).cpu().numpy())
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
        with device_guard(), torch.no_grad():
            xq = imaging.uint8_to_tensor(np.asarray(arr[query_id])).unsqueeze(0).to(self.device)
            zq = self.model.encode(xq).cpu().numpy()[0]
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
        xa = imaging.uint8_to_tensor(np.asarray(arr[a_id])).unsqueeze(0).to(self.device)
        xb = imaging.uint8_to_tensor(np.asarray(arr[b_id])).unsqueeze(0).to(self.device)
        # En la U-Net el cuello z casi no lleva información (los skips lo "puentean"): decodificar
        # un z suelto sin skips produce un blob gris, no una cara. Para que la interpolación sea
        # un morphing real, interpolamos el código COMPLETO de esa variante (cuello + skips) y
        # decodificamos con los skips interpolados. En básico/grande el código es solo z.
        use_skips = hasattr(self.model, "encode_with_skips")
        with device_guard(), torch.no_grad():
            if use_skips:
                za, skips_a = self.model.encode_with_skips(xa)
                zb, skips_b = self.model.encode_with_skips(xb)
            else:
                za, zb = self.model.encode(xa), self.model.encode(xb)
            frames = []
            for s in range(steps):
                alpha = s / (steps - 1)
                z = (1 - alpha) * za + alpha * zb
                if use_skips:
                    skips = [(1 - alpha) * sa + alpha * sb for sa, sb in zip(skips_a, skips_b)]
                    img = self.model.decode(z, skips)[0]
                else:
                    img = self.model.decode(z)[0]
                frames.append({"alpha": round(alpha, 3), "image": imaging.tensor_to_b64(img)})
        return {"a_id": a_id, "b_id": b_id, "steps": steps, "frames": frames}


# Singleton compartido por los routers.
ae_service = AEService()


if __name__ == "__main__":
    # Genera el checkpoint demo de LAS TRES variantes:  python -m app.services.ae_service
    # Cada una entrena en modo full acotado (DEMO_EPOCHS) y se guarda en ae_<arch>_demo.pt.
    summary: list[dict] = []
    for arch in ARCHS:
        svc = AEService()
        print(f"[ae] === variante '{arch}' · full {DEMO_EPOCHS} epochs en {svc.device.type} ===", flush=True)
        hp = AEHyperParams(epochs=DEMO_EPOCHS, batch_size=256, arch=arch)
        for ev in svc.iter_train("full", hp, seed=42):
            if ev["type"] == "epoch":
                print(f"[ae]   [{arch}] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}", flush=True)
            elif ev["type"] == "done":
                print(f"[ae]   [{arch}] listo · loss final {ev['loss']:.5f}", flush=True)
        path = demo_path(arch)
        svc.save_checkpoint(path)
        print(f"[ae]   [{arch}] checkpoint guardado en {path}", flush=True)
        m = svc.metrics()
        print(f"[ae]   [{arch}] held-out (n={m['n']}): "
              f"MSE {m['mse']:.5f} · PSNR {m['psnr']:.2f} dB · SSIM {m['ssim']:.4f} "
              f"· params {count_params(svc.model) / 1e6:.2f}M", flush=True)
        summary.append({"arch": arch, **{k: m[k] for k in ("mse", "psnr", "ssim")},
                        "params": count_params(svc.model)})

    print("\n[ae] === Resumen de variantes (held-out fijo) ===", flush=True)
    print(f"[ae] {'arch':8s} {'params':>10s} {'MSE':>10s} {'PSNR(dB)':>10s} {'SSIM':>8s}", flush=True)
    for r in summary:
        print(f"[ae] {r['arch']:8s} {r['params'] / 1e6:9.2f}M {r['mse']:10.5f} "
              f"{r['psnr']:10.2f} {r['ssim']:8.4f}", flush=True)
