"""Modelo de difusión DDPM pequeño (CLAUDE.md §4.4).

Un modelo de difusión NO tiene encoder: aprende a GENERAR partiendo de ruido puro y
limpiándolo paso a paso. Aquí implementamos:

1. Una UNet PEQUEÑA condicionada en el tiempo `t` (embedding sinusoidal) que, dada una
   imagen ruidosa `x_t` y el paso `t`, predice el ruido `ε` que se le añadió.
2. El proceso DDPM (Ho et al., 2020):
   - schedule lineal de `β_t`, y de ahí `α_t = 1−β_t` y `ᾱ_t = ∏ α_s` (alphas_cumprod).
   - forward `q(x_t | x_0)`: `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`   (`q_sample`).
   - reverse `p(x_{t-1} | x_t)`: un paso de denoising que usa el ε predicho (`p_sample`),
     y el bucle completo de muestreo (`sample_loop`).

RENDIMIENTO (MPS/CPU): trabajamos a 32×32×3 con una UNet de pocos canales y `T=200`. El
muestreo puede submuestrear pasos (p. ej. 50) para ser interactivo. La parametrización y
la varianza del paso reverse son las estándar de DDPM (varianza fija = β_t).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

IMG_CH = 3
IMG_SIZE = 32          # resolución interna del modelo (downsample del dataset 64→32)
BASE = 64              # canales base de la UNet (64→128→128 por el camino down)
TIME_DIM = 128         # dimensión del embedding temporal

# ---- schedule DDPM (constantes del módulo) -------------------------------------------
TIMESTEPS = 200        # T: nº de pasos del proceso de difusión
BETA_START = 1e-4
BETA_END = 0.02
# Schedule por defecto. "cosine" (Nichol & Dhariwal 2021) destruye el detalle más despacio
# que el lineal y mejora notablemente la calidad de muestreo a pocos pasos/resoluciones bajas.
DEFAULT_SCHEDULE = "cosine"
COSINE_S = 0.008       # pequeño offset que evita β_t≈0 cerca de t=0 (paper)


def make_beta_schedule_linear(timesteps: int = TIMESTEPS) -> torch.Tensor:
    """Schedule lineal de β_t entre BETA_START y BETA_END (Ho et al., 2020)."""
    return torch.linspace(BETA_START, BETA_END, timesteps, dtype=torch.float32)


def make_beta_schedule_cosine(timesteps: int = TIMESTEPS, s: float = COSINE_S) -> torch.Tensor:
    """Schedule cosine (Nichol & Dhariwal 2021).

    Define ᾱ_t = cos²(((t/T)+s)/(1+s) · π/2), normaliza para que ᾱ_0 = 1, y deriva
    β_t = 1 − ᾱ_t/ᾱ_{t-1}, recortado a [1e-4, 0.999] para estabilidad numérica.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    acp = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    acp = acp / acp[0]
    betas = 1.0 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-4, 0.999).to(torch.float32)


def make_beta_schedule(timesteps: int = TIMESTEPS, schedule: str = DEFAULT_SCHEDULE) -> torch.Tensor:
    """Devuelve el schedule de β_t pedido ("cosine" por defecto, o "linear")."""
    if str(schedule).lower() == "linear":
        return make_beta_schedule_linear(timesteps)
    return make_beta_schedule_cosine(timesteps)


class DiffusionSchedule:
    """Precalcula y aloja en un device los coeficientes derivados del schedule.

    Mantener todo en tensores (en el device) evita recomputar raíces en cada paso y hace
    el forward/reverse triviales de leer. Se reconstruye si cambia `timesteps`.
    """

    def __init__(
        self,
        timesteps: int = TIMESTEPS,
        device: torch.device | None = None,
        schedule: str = DEFAULT_SCHEDULE,
    ) -> None:
        self.timesteps = int(timesteps)
        self.schedule = str(schedule)
        dev = device or torch.device("cpu")
        betas = make_beta_schedule(self.timesteps, self.schedule).to(dev)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        # coeficientes para q_sample y p_sample
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        # varianza del paso reverse (fija = β_t, la opción "small" de DDPM)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        for name, val in list(self.__dict__.items()):
            if isinstance(val, torch.Tensor):
                setattr(self, name, val.to(device))
        return self


def _gather(coef: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """Toma coef[t] por muestra del batch y lo reestructura a (B,1,1,1) para difundir."""
    out = coef.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(shape) - 1)))


# ---- embedding temporal sinusoidal ---------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Embedding sinusoidal del paso `t` (como en Transformers/DDPM).

    t: (B,) enteros del paso; devuelve (B, dim) con senos/cosenos a varias frecuencias,
    de modo que la red "sabe" en qué punto del proceso de ruido está.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # dim impar: rellena una columna
        emb = F.pad(emb, (0, 1))
    return emb


# ---- bloques de la UNet --------------------------------------------------------------
class ResBlock(nn.Module):
    """Bloque residual con inyección del embedding temporal (FiLM-additivo)."""

    def __init__(self, cin: int, cout: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Down(nn.Module):
    """Bloque residual + downsample ×2 (stride-2 conv)."""

    def __init__(self, cin: int, cout: int, time_dim: int) -> None:
        super().__init__()
        self.block = ResBlock(cin, cout, time_dim)
        self.down = nn.Conv2d(cout, cout, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x, t_emb)   # para la skip connection (antes de bajar resolución)
        return self.down(skip), skip


class Up(nn.Module):
    """Upsample ×2 + concat de la skip + bloque residual.

    `in_ch`: canales de la entrada (del nivel inferior); `skip_ch`: canales de la skip que
    se concatena; `out_ch`: canales de salida del bloque.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.block = ResBlock(in_ch + skip_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class TimeUNet(nn.Module):
    """UNet pequeña condicionada en `t` para predecir el ruido ε de imágenes 32×32×3.

    Camino: 32→16→8 (down), bottleneck en 8×8, y 8→16→32 (up) con skip connections.
    Pocos canales (BASE=64) para que sea viable en MPS/CPU. Las skip connections llevan los
    canales de salida de cada bloque down: s1 → c2, s2 → c3.
    """

    def __init__(self) -> None:
        super().__init__()
        c1, c2, c3 = BASE, BASE * 2, BASE * 2  # 64, 128, 128
        # MLP del embedding temporal compartido por todos los bloques
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, TIME_DIM),
            nn.SiLU(),
            nn.Linear(TIME_DIM, TIME_DIM),
        )
        self.in_conv = nn.Conv2d(IMG_CH, c1, 3, padding=1)   # 32×32, c1
        self.down1 = Down(c1, c2, TIME_DIM)                  # 32 -> 16 (skip s1: c2)
        self.down2 = Down(c2, c3, TIME_DIM)                  # 16 -> 8  (skip s2: c3)
        # bottleneck 8×8 con DOS bloques residuales: aumento de capacidad barato (resolución
        # más baja) que ayuda a la calidad sin penalizar mucho el coste de muestreo.
        self.mid1 = ResBlock(c3, c3, TIME_DIM)
        self.mid2 = ResBlock(c3, c3, TIME_DIM)
        self.up1 = Up(c3, c3, c2, TIME_DIM)                  # 8  -> 16, concat s2(c3) -> c2
        self.up2 = Up(c2, c2, c1, TIME_DIM)                  # 16 -> 32, concat s1(c2) -> c1
        self.out_norm = nn.GroupNorm(8, c1)
        self.out_conv = nn.Conv2d(c1, IMG_CH, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, TIME_DIM))
        h0 = self.in_conv(x)
        h1, s1 = self.down1(h0, t_emb)
        h2, s2 = self.down2(h1, t_emb)
        m = self.mid1(h2, t_emb)
        m = self.mid2(m, t_emb)
        u = self.up1(m, s2, t_emb)
        u = self.up2(u, s1, t_emb)
        return self.out_conv(F.silu(self.out_norm(u)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---- EMA de pesos (clave para la calidad de muestreo) --------------------------------
class EMA:
    """Media móvil exponencial de los parámetros del modelo.

    Mantiene una copia "shadow" de los pesos que se actualiza en cada step de entrenamiento
    como  θ_ema ← decay·θ_ema + (1−decay)·θ. Muestrear con los pesos EMA (en lugar de los
    pesos "crudos" que oscilan con el último minibatch) es una de las prácticas que MÁS
    mejora la calidad visual en DDPM, sin coste de entrenamiento extra apreciable.

    Uso:
        ema = EMA(model, decay=0.999)
        ...                       # tras cada opt.step():
        ema.update(model)
        ...                       # para muestrear con los pesos suavizados:
        ema.copy_to(shadow_model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        # clones desacoplados del grafo, en el mismo device que el modelo. Solo seguimos los
        # tensores flotantes (parámetros/buffers float); esta UNet no tiene buffers enteros
        # (sin BatchNorm), así que `copy_to` reconstruye el modelo de muestreo por completo.
        self.shadow: dict[str, torch.Tensor] = {
            name: p.detach().clone()
            for name, p in model.state_dict().items()
            if p.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for name, shadow in self.shadow.items():
            new = msd[name].detach()
            if shadow.device != new.device:  # robustez si el modelo se movió de device
                shadow = shadow.to(new.device)
                self.shadow[name] = shadow
            # lerp in-place:  shadow = d·shadow + (1−d)·new
            shadow.mul_(d).add_(new, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """Vuelca los pesos EMA en `model` in-place (para muestrear con ellos)."""
        msd = model.state_dict()
        for name, shadow in self.shadow.items():
            msd[name].copy_(shadow.to(msd[name].device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        """state_dict de los pesos EMA (mismo formato que el del modelo, en CPU)."""
        return {name: p.detach().cpu().clone() for name, p in self.shadow.items()}

    def to(self, device: torch.device) -> "EMA":
        for name in list(self.shadow.keys()):
            self.shadow[name] = self.shadow[name].to(device)
        return self


# ---- proceso DDPM: forward y reverse -------------------------------------------------
def q_sample(
    sched: DiffusionSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    """Forward `q(x_t | x_0)`: añade ruido al instante t en un solo paso.

    x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε
    """
    sqrt_acp = _gather(sched.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_om = _gather(sched.sqrt_one_minus_alphas_cumprod, t, x0.shape)
    return sqrt_acp * x0 + sqrt_om * noise


@torch.no_grad()
def p_sample(
    model: TimeUNet, sched: DiffusionSchedule, x: torch.Tensor, t: torch.Tensor, t_index: int
) -> torch.Tensor:
    """Un paso del reverse `p(x_{t-1} | x_t)` usando el ε predicho por la red.

    media = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ);  se añade ruido salvo en el último paso.
    """
    betas_t = _gather(sched.betas, t, x.shape)
    sqrt_om = _gather(sched.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip = _gather(sched.sqrt_recip_alphas, t, x.shape)

    eps = model(x, t)
    mean = sqrt_recip * (x - betas_t / sqrt_om * eps)
    if t_index == 0:
        return mean
    var = _gather(sched.posterior_variance, t, x.shape)
    return mean + torch.sqrt(var) * torch.randn_like(x)


def sampling_timesteps(total: int, steps: int) -> list[int]:
    """Submuestrea `steps` índices de tiempo de [0, total) de forma uniforme y descendente.

    Devuelve los índices de mayor (más ruido) a menor (imagen limpia). Con steps==total se
    recorre todo el proceso DDPM; con menos, se salta pasos (más rápido, algo menos fino).
    """
    steps = max(1, min(int(steps), total))
    idx = torch.linspace(0, total - 1, steps).round().long().tolist()
    uniq = sorted(set(int(i) for i in idx), reverse=True)
    return uniq


@torch.no_grad()
def sample_loop(
    model: TimeUNet,
    sched: DiffusionSchedule,
    n: int,
    device: torch.device,
    steps: int | None = None,
    capture: list[int] | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Muestrea `n` imágenes desde ruido puro hacia imagen limpia con **DDIM** (Song et al.,
    2021), determinista (η=0).

    Por qué DDIM y no el reverse DDPM ancestral: el paso DDPM clásico solo avanza de t a t−1.
    Si submuestreamos (p. ej. 50 de 200 pasos) y aplicamos el paso de un solo t, NO se quita
    suficiente ruido y la salida queda ruidosa. DDIM, en cambio, salta correctamente entre
    timesteps arbitrarios reusando la predicción de x₀, así que funciona con pocos pasos.

    En cada paso: ε̂ = modelo(x_t, t); x̂₀ = (x_t − √(1−ᾱ_t)·ε̂)/√ᾱ_t (recortado a [−1,1]);
    x_{t_prev} = √ᾱ_{t_prev}·x̂₀ + √(1−ᾱ_{t_prev})·ε̂.   En el último paso ᾱ_prev=1 → x = x̂₀.

    - `steps`: nº de pasos de muestreo (submuestreo de T). None = todos.
    - `capture`: POSICIONES en la secuencia donde guardar una instantánea (trayectoria).

    Devuelve (x_final, snapshots), todos en [0,1] (B,C,H,W).
    """
    was_training = model.training
    model.eval()
    ts = sampling_timesteps(sched.timesteps, steps or sched.timesteps)  # descendente
    capture_set = set(capture or [])
    acp = sched.alphas_cumprod

    x = torch.randn(n, IMG_CH, IMG_SIZE, IMG_SIZE, device=device)
    snapshots: list[torch.Tensor] = []
    if 0 in capture_set:
        snapshots.append(_to_img(x))

    for pos, t_idx in enumerate(ts):
        t = torch.full((n,), t_idx, device=device, dtype=torch.long)
        eps = model(x, t)
        acp_t = acp[t_idx]
        x0 = ((x - torch.sqrt(1.0 - acp_t) * eps) / torch.sqrt(acp_t)).clamp(-1.0, 1.0)
        t_next = ts[pos + 1] if pos + 1 < len(ts) else -1
        acp_prev = acp[t_next] if t_next >= 0 else torch.ones((), device=device)
        x = torch.sqrt(acp_prev) * x0 + torch.sqrt(1.0 - acp_prev) * eps
        if (pos + 1) in capture_set:
            snapshots.append(_to_img(x))

    if was_training:
        model.train()
    return _to_img(x), snapshots


def _to_img(x: torch.Tensor) -> torch.Tensor:
    """Las imágenes viven en [-1,1] durante el proceso; las llevamos a [0,1] para mostrar."""
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)
