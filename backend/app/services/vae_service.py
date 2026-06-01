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

from .. import imaging
from ..data import dataset
from ..device import get_device
from ..models.vae import ConvVAE, count_params
from ..seeding import set_seed

DEMO_PATH = Path(__file__).resolve().parents[1] / "checkpoints" / "vae_demo.pt"

QUICK_N = 6000          # subconjunto para el modo "quick"
QUICK_MAX_EPOCHS = 6
DEMO_EPOCHS = 12        # checkpoint demo (modo full)
LOG_EVERY = 25          # steps entre eventos de progreso
PREVIEW_IDS = [12, 800, 4096, 20000]  # caras fijas para la vista previa
EMBED_SAMPLE = 1200     # nº de z muestreados para el mapa latente / vecinos / clusters


def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) ‖ N(0,I)) promediada por muestra del lote.

    KL = -½ · mean( sum_j (1 + logσ²_j − μ_j² − exp(logσ²_j)) ).
    Mide cuánto se aleja el posterior del prior gaussiano estándar.
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

    def sanitized(self) -> "VAEHyperParams":
        return VAEHyperParams(
            latent_dim=int(max(8, min(512, self.latent_dim))),
            learning_rate=float(max(1e-5, min(1e-1, self.learning_rate))),
            epochs=int(max(1, min(200, self.epochs))),
            loss="l1" if str(self.loss).lower() == "l1" else "mse",
            beta=float(max(0.0, min(10.0, self.beta))),
            batch_size=int(max(16, min(512, self.batch_size))),
        )


class VAEService:
    """Mantiene el modelo VAE y su estado entre peticiones."""

    def __init__(self) -> None:
        self.device = get_device()
        self.model: ConvVAE | None = None
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

    def _preview(self, model: ConvVAE) -> list[str]:
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
    def iter_train(self, mode: str, hp: VAEHyperParams, seed: int) -> Iterator[dict]:
        """Entrena emitiendo eventos {type, epoch, step, loss, recon_loss, kl_loss, preview?}.

        La pérdida es `recon + β·KL`. En cada epoch se emiten también `recon_loss` y `kl_loss`
        por separado para poder graficar el trade-off reconstrucción ↔ regularidad.
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
            epochs = min(hp.epochs, QUICK_MAX_EPOCHS) if quick else hp.epochs

            model = ConvVAE(hp.latent_dim).to(self.device)
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
            yield {"type": "start", "epochs": epochs, "n": int(n_use), "mode": mode,
                   "hyperparams": asdict(hp), "seed": seed}

            for epoch in range(1, epochs + 1):
                rng.shuffle(indices)
                run_total = torch.zeros((), device=self.device)
                run_recon = torch.zeros((), device=self.device)
                run_kl = torch.zeros((), device=self.device)
                nb = 0
                for batch in self._iter_batches(indices, hp.batch_size):
                    opt.zero_grad(set_to_none=True)
                    recon, mu, logvar, _ = model(batch)
                    recon_loss = recon_crit(recon, batch)
                    kl_loss = _kl_divergence(mu, logvar)
                    loss = recon_loss + hp.beta * kl_loss
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
                       "preview": self._preview(model)}

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
            yield {"type": "done", "epochs": epochs,
                   "loss": history[-1]["loss"] if history else None,
                   "loss_history": history}
        finally:
            self.training = False

    def start_training(self, mode: str, hp: VAEHyperParams, seed: int) -> "queue.Queue":
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

        threading.Thread(target=worker, name="vae-train", daemon=True).start()
        return q

    # ---------- búsqueda en rejilla (grid search) ----------
    def _eval_mse(self, model: ConvVAE, ids: np.ndarray) -> float:
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
                model = ConvVAE(ld).to(self.device)
                model.train()
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                recon_crit: nn.Module = nn.L1Loss() if ls == "l1" else nn.MSELoss()
                ids = train_ids.copy()
                for ep in range(epochs):
                    np.random.default_rng(seed + ep + 1).shuffle(ids)
                    for batch in self._iter_batches(ids, 256):
                        opt.zero_grad(set_to_none=True)
                        recon, mu, logvar, _ = model(batch)
                        loss = recon_crit(recon, batch) + beta * _kl_divergence(mu, logvar)
                        loss.backward()
                        opt.step()
                        if self._cancel:
                            yield {"type": "cancelled"}
                            return
                    yield {"type": "progress", "index": idx, "total": total,
                           "epoch": ep + 1, "epochs": epochs}
                cfg = {"latent_dim": ld, "learning_rate": lr, "loss": ls, "epochs": epochs}
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
        self.hp = VAEHyperParams(**ckpt["hyperparams"]).sanitized()
        model = ConvVAE(self.hp.latent_dim).to(self.device)
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
    # Genera el checkpoint demo:  python -m app.services.vae_service
    svc = VAEService()
    print(f"[vae] entrenando checkpoint demo (full, {DEMO_EPOCHS} epochs, beta=1.0) en {svc.device.type}…", flush=True)
    for ev in svc.iter_train("full", VAEHyperParams(epochs=DEMO_EPOCHS, beta=1.0, batch_size=256), seed=42):
        if ev["type"] == "epoch":
            print(
                f"[vae]   epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}"
                f" (recon {ev['recon_loss']:.5f} · kl {ev['kl_loss']:.5f})",
                flush=True,
            )
        elif ev["type"] == "done":
            print(f"[vae] listo · loss final {ev['loss']:.5f}", flush=True)
    svc.save_checkpoint(DEMO_PATH)
    print(f"[vae] checkpoint demo guardado en {DEMO_PATH}", flush=True)
