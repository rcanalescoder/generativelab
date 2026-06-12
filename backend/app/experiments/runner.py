"""Runner del ciclo v3: entrena una config, la evalúa con el protocolo fijo (quality.py) y
registra el run en experiments/ledger.jsonl (+ LEADERBOARD.md + grid PNG). Headless:

  .venv/bin/python -m app.experiments.runner --model vae --tag baseline-basico \
      --mode quick --hp '{"epochs": 12}' --eval-n 2048

  --model       vae | gan | diffusion
  --tag         etiqueta humana del experimento (entra en el run_id)
  --mode        quick | full (quick usa el subconjunto QUICK_N del servicio)
  --train-n     override del tamaño del subconjunto (solo afecta en quick / GAN)
  --hp          JSON con hiperparámetros (se mezclan sobre los defaults del servicio)
  --seed        semilla de entrenamiento (default 42)
  --eval-n      nº de muestras generadas para evaluar (default 2048; diffusion: usar 512)
  --steps       pasos de muestreo DDIM (solo diffusion; default 60)
  --early-stop  activa el early stopping del servicio (default: OFF, reproducibilidad)
  --save-ckpt   ruta donde guardar el checkpoint tras entrenar (opcional)
  --load-demo   en vez de entrenar, carga el checkpoint demo de esa arch y solo evalúa
  --notes       nota libre para el ledger

Un entrenamiento MPS a la vez. No toca el estado del servidor (instancia services propios).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.nn import functional as F

from . import quality


# ---------------------------------------------------------------------------
# Muestreo por modelo (todas devuelven NCHW [0,1] en CPU, a 64×64 para mostrar)
# ---------------------------------------------------------------------------
def _sample_vae(svc, n: int, seed: int, batch: int = 256) -> torch.Tensor:
    from ..seeding import set_seed

    set_seed(seed)
    svc.model.eval()
    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, svc.model.latent_dim, device=svc.device)
            outs.append(svc.model.decode(z).cpu())
    return torch.cat(outs, 0)


def _sample_gan(svc, n: int, seed: int, batch: int = 256) -> torch.Tensor:
    from ..models.gan import to_image_range

    g = torch.Generator(device="cpu").manual_seed(seed)
    svc.generator.eval()
    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            z = torch.randn(b, svc.hp.z_dim, generator=g).to(svc.device)
            outs.append(to_image_range(svc.generator(z)).cpu())
    return torch.cat(outs, 0)


def _sample_diffusion(svc, n: int, seed: int, steps: int, batch: int = 64) -> torch.Tensor:
    from ..models.diffusion import sample_loop
    from ..seeding import set_seed

    set_seed(seed)
    svc._ensure_schedule(svc.hp.timesteps)
    outs: list[torch.Tensor] = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        imgs, _ = sample_loop(svc.model, svc.sched, b, svc.device, steps=steps)
        if imgs.shape[-1] != 64:
            imgs = F.interpolate(imgs, size=64, mode="bilinear", align_corners=False)
        outs.append(imgs.cpu())
        print(f"[runner]   muestreo {min(i + b, n)}/{n}", flush=True)
    return torch.cat(outs, 0)


# ---------------------------------------------------------------------------
# Construcción de servicio + hiperparámetros por modelo
# ---------------------------------------------------------------------------
def _build(model: str, hp_json: dict, train_n: int | None):
    if model == "vae":
        import app.services.vae_service as mod

        svc = mod.VAEService()
        if train_n:
            mod.QUICK_N = train_n
        hp = replace(mod.VAEHyperParams(), **{k: v for k, v in hp_json.items() if hasattr(mod.VAEHyperParams(), k)})
        return svc, hp, mod
    if model == "gan":
        import app.services.gan_service as mod

        svc = mod.GANService()
        if train_n:
            mod.QUICK_N = train_n
        hp = replace(mod.GANHyperParams(), **{k: v for k, v in hp_json.items() if hasattr(mod.GANHyperParams(), k)})
        return svc, hp, mod
    if model == "diffusion":
        import app.services.diffusion_service as mod

        svc = mod.DiffusionService()
        if train_n:
            mod.QUICK_N = train_n
        hp = replace(
            mod.DiffusionHyperParams(),
            **{k: v for k, v in hp_json.items() if hasattr(mod.DiffusionHyperParams(), k)},
        )
        return svc, hp, mod
    raise SystemExit(f"modelo desconocido: {model}")


def _train(model: str, svc, hp, mode: str, seed: int, early_stop: bool) -> dict:
    """Itera el iter_train del servicio imprimiendo el progreso. Devuelve resumen."""
    t0 = time.time()
    epochs_run = 0
    final: dict = {}
    if model == "gan":
        events = svc.iter_train(mode, hp, seed)
    else:
        events = svc.iter_train(mode, hp, seed, early_stop=early_stop)
    for ev in events:
        if ev["type"] == "epoch":
            epochs_run = ev["epoch"]
            if model == "gan":
                print(f"[runner] epoch {ev['epoch']}/{ev['epochs']} · g {ev['g_loss']:.4f} · d {ev['d_loss']:.4f}", flush=True)
            elif model == "vae":
                print(
                    f"[runner] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f} "
                    f"(recon {ev['recon_loss']:.5f} · kl {ev['kl_loss']:.5f})",
                    flush=True,
                )
            else:
                print(f"[runner] epoch {ev['epoch']}/{ev['epochs']} · loss {ev['loss']:.5f}", flush=True)
        elif ev["type"] == "error":
            raise SystemExit(f"error de entrenamiento: {ev['message']}")
        elif ev["type"] == "done":
            final = ev
    return {
        "epochs_run": epochs_run,
        "train_seconds": round(time.time() - t0, 1),
        "final": {k: v for k, v in final.items() if k in ("loss", "g_loss", "d_loss", "stopped_early")},
    }


def _rebuild_leaderboard() -> None:
    """Regenera experiments/LEADERBOARD.md desde el ledger (por modelo, KID ascendente)."""
    if not quality.LEDGER_PATH.exists():
        return
    rows = [json.loads(l) for l in quality.LEDGER_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    out = ["# Leaderboard v3 (KID×1000 ↓ = mejor)\n"]
    for model in ("vae", "gan", "diffusion"):
        if model not in by_model:
            continue
        out.append(f"\n## {model.upper()}\n")
        out.append("| Run | Tag | KID×1000 | Diversidad (ratio) | Epochs | Train (s) | Grid |")
        out.append("|---|---|---|---|---|---|---|")
        for r in sorted(by_model[model], key=lambda r: r["metrics"].get("kid_x1000", 1e9)):
            m = r["metrics"]
            grid = Path(m.get("grid", "")).name
            out.append(
                f"| {r['run_id']} | {r['tag']} | {m.get('kid_x1000', '—')} ± {m.get('kid_std_x1000', '—')} "
                f"| {m.get('diversity_ratio', '—')} | {r.get('epochs_run', '—')} "
                f"| {r.get('train_seconds', '—')} | {grid} |"
            )
    (quality.EXP_DIR / "LEADERBOARD.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["vae", "gan", "diffusion"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="quick", choices=["quick", "full"])
    ap.add_argument("--train-n", type=int, default=None)
    ap.add_argument("--hp", default="{}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-n", type=int, default=2048)
    ap.add_argument("--eval-seed", type=int, default=9999)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--save-ckpt", default=None)
    ap.add_argument("--load-demo", default=None, metavar="ARCH")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    hp_json = json.loads(args.hp)
    svc, hp, _mod = _build(args.model, hp_json, args.train_n)
    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{args.model}-{args.tag}"
    print(f"[runner] run {run_id} · device {svc.device.type} · hp {hp_json}", flush=True)

    train_info: dict = {}
    if args.load_demo:
        if not svc.load_demo(args.load_demo):
            raise SystemExit(f"no se pudo cargar el demo '{args.load_demo}'")
        print(f"[runner] demo '{args.load_demo}' cargado (sin entrenar)", flush=True)
    else:
        train_info = _train(args.model, svc, hp, args.mode, args.seed, args.early_stop)

    # --- muestreo + evaluación con el protocolo fijo ---
    n = args.eval_n
    if args.model == "vae":
        fake = _sample_vae(svc, n, args.eval_seed)
    elif args.model == "gan":
        fake = _sample_gan(svc, n, args.eval_seed)
    else:
        fake = _sample_diffusion(svc, n, args.eval_seed, args.steps)
    print(f"[runner] {fake.shape[0]} muestras generadas; evaluando…", flush=True)

    grid_path = quality.GRIDS_DIR / f"{run_id}.png"
    metrics = quality.evaluate_generation(fake, grid_path=grid_path)

    recon = None
    if args.model == "vae":
        m = svc.metrics(n=256, n_examples=0)
        recon = {k: round(float(m[k]), 5) for k in ("mse", "psnr", "ssim")}

    ckpt = None
    if args.save_ckpt:
        ckpt_path = Path(args.save_ckpt)
        svc.save_checkpoint(ckpt_path)
        ckpt = str(ckpt_path)
        print(f"[runner] checkpoint guardado en {ckpt_path}", flush=True)

    entry = {
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": args.model,
        "tag": args.tag,
        "mode": args.mode,
        "train_n": args.train_n,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "hp": asdict(hp) | ({"sample_steps": args.steps} if args.model == "diffusion" else {}),
        **train_info,
        "metrics": metrics,
        "recon": recon,
        "ckpt": ckpt,
        "notes": args.notes,
    }
    quality.EXP_DIR.mkdir(parents=True, exist_ok=True)
    with quality.LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    _rebuild_leaderboard()
    print("[runner] RESULT " + json.dumps(entry, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
