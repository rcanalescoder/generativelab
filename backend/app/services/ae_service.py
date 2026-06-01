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

from .. import imaging
from ..data import dataset
from ..device import get_device
from ..models.ae import ConvAutoencoder, count_params
from ..seeding import set_seed

CKPT_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "ae_demo.pt"

QUICK_N = 6000          # subconjunto para el modo "quick"
QUICK_MAX_EPOCHS = 6
DEMO_EPOCHS = 10        # checkpoint demo (modo full)
LOG_EVERY = 25          # steps entre eventos de progreso
PREVIEW_IDS = [12, 800, 4096, 20000]  # caras fijas para la vista previa
EMBED_SAMPLE = 1200     # nº de z muestreados para el mapa latente / vecinos / clusters


@dataclass
class AEHyperParams:
    latent_dim: int = 128
    learning_rate: float = 1e-3
    epochs: int = 30
    loss: str = "mse"  # "mse" | "l1"
    batch_size: int = 256

    def sanitized(self) -> "AEHyperParams":
        return AEHyperParams(
            latent_dim=int(max(8, min(512, self.latent_dim))),
            learning_rate=float(max(1e-5, min(1e-1, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            loss="l1" if str(self.loss).lower() == "l1" else "mse",
            batch_size=int(max(16, min(512, self.batch_size))),
        )


class AEService:
    """Mantiene el modelo AE y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.model: ConvAutoencoder | None = None
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

    def _preview(self, model: ConvAutoencoder) -> list[str]:
        was_training = model.training
        model.eval()
        arr = dataset.load_array()
        n = dataset.count()
        ids = [i for i in PREVIEW_IDS if i < n][:4]
        with torch.no_grad():
            xs = torch.stack([imaging.uint8_to_tensor(np.asarray(arr[i])) for i in ids]).to(self.device)
            xh, _ = model(xs)
        if was_training:
            model.train()
        return [imaging.tensor_to_b64(xh[k]) for k in range(len(ids))]

    # ---------- entrenamiento ----------
    def iter_train(self, mode: str, hp: AEHyperParams, seed: int) -> Iterator[dict]:
        """Entrena emitiendo eventos {type, epoch, step, loss, preview?}. Guarda checkpoint al final."""
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
            epochs = min(hp.epochs, QUICK_MAX_EPOCHS) if quick else hp.epochs

            model = ConvAutoencoder(hp.latent_dim).to(self.device)
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
            yield {"type": "start", "epochs": epochs, "n": int(n_use), "mode": mode,
                   "hyperparams": asdict(hp), "seed": seed}

            for epoch in range(1, epochs + 1):
                rng.shuffle(indices)
                running = torch.zeros((), device=self.device)
                nb = 0
                for batch in self._iter_batches(indices, hp.batch_size):
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

            # finalizar: adoptar el modelo entrenado y guardar
            model.eval()
            self.model = model
            self.hp = hp
            self.seed = seed
            self.trained = True
            self.outdated = False
            self.loss_history = history
            self._invalidate_embedding()
            self.save_checkpoint()
            yield {"type": "done", "epochs": epochs, "loss": history[-1]["loss"] if history else None,
                   "loss_history": history}
        finally:
            self.training = False

    def start_training(self, mode: str, hp: AEHyperParams, seed: int) -> "queue.Queue":
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
                for ev in self.iter_train(mode, hp, seed):
                    q.put(ev)
            except Exception as exc:  # pragma: no cover - defensivo
                q.put({"type": "error", "message": str(exc)})
            finally:
                q.put(None)

        threading.Thread(target=worker, name="ae-train", daemon=True).start()
        return q

    # ---------- checkpoint ----------
    def save_checkpoint(self) -> None:
        if self.model is None:
            return
        CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hyperparams": asdict(self.hp),
                "seed": self.seed,
                "loss_history": self.loss_history,
                "version": 1,
            },
            CKPT_PATH,
        )

    def load_checkpoint(self) -> bool:
        if not CKPT_PATH.exists():
            return False
        ckpt = torch.load(CKPT_PATH, map_location=self.device)
        self.hp = AEHyperParams(**ckpt["hyperparams"]).sanitized()
        model = ConvAutoencoder(self.hp.latent_dim).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self.model = model
        self.seed = ckpt.get("seed", 42)
        self.loss_history = ckpt.get("loss_history", [])
        self.trained = True
        self.outdated = False
        self._invalidate_embedding()
        return True

    # ---------- reconstrucción ----------
    def reconstruct(self, ids: list[int]) -> list[dict]:
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
                recon, _ = self.model(x.unsqueeze(0))
                recon = recon[0]
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
        with torch.no_grad():
            recon = self.model(x.unsqueeze(0).to(self.device))[0][0]
        return {
            "original": imaging.tensor_to_b64(x),
            "reconstruction": imaging.tensor_to_b64(recon),
            "diff": imaging.diff_b64(x.cpu(), recon.cpu()),
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
        with torch.no_grad():
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
        with torch.no_grad():
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
        with torch.no_grad():
            za = self.model.encode(imaging.uint8_to_tensor(np.asarray(arr[a_id])).unsqueeze(0).to(self.device))
            zb = self.model.encode(imaging.uint8_to_tensor(np.asarray(arr[b_id])).unsqueeze(0).to(self.device))
            frames = []
            for s in range(steps):
                alpha = s / (steps - 1)
                img = self.model.decode((1 - alpha) * za + alpha * zb)[0]
                frames.append({"alpha": round(alpha, 3), "image": imaging.tensor_to_b64(img)})
        return {"a_id": a_id, "b_id": b_id, "steps": steps, "frames": frames}


# Singleton compartido por los routers.
ae_service = AEService()


if __name__ == "__main__":
    # Genera el checkpoint demo:  python -m app.services.ae_service
    svc = AEService()
    print(f"[ae] entrenando checkpoint demo (full, {DEMO_EPOCHS} epochs) en {svc.device.type}…", flush=True)
    for ev in svc.iter_train("full", AEHyperParams(epochs=DEMO_EPOCHS, batch_size=256), seed=42):
        if ev["type"] == "epoch":
            print(f"[ae]   epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}", flush=True)
        elif ev["type"] == "done":
            print(f"[ae] listo · loss final {ev['loss']:.5f} · guardado en {CKPT_PATH}", flush=True)
