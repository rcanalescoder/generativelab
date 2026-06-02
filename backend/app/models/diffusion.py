"""Modelo de difusión DDPM (CLAUDE.md §4.4) con DOS variantes elegibles.

Un modelo de difusión NO tiene encoder: aprende a GENERAR partiendo de ruido puro y
limpiándolo paso a paso. Aquí implementamos:

1. Una UNet condicionada en el tiempo `t` (embedding sinusoidal) que, dada una imagen
   ruidosa `x_t` y el paso `t`, predice el ruido `ε` que se le añadió. La UNet es
   PARAMETRIZABLE (`TimeUNet(img_size, base, ch_mults, attn_res)`): cambiando esos
   argumentos obtenemos arquitecturas de distinto tamaño/resolución.
2. El proceso DDPM (Ho et al., 2020):
   - schedule lineal de `β_t`, y de ahí `α_t = 1−β_t` y `ᾱ_t = ∏ α_s` (alphas_cumprod).
   - forward `q(x_t | x_0)`: `x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε`   (`q_sample`).
   - reverse `p(x_{t-1} | x_t)`: un paso de denoising que usa el ε predicho (`p_sample`),
     y el bucle completo de muestreo **DDIM** (`sample_loop`).

VARIANTES (`build_diffusion(arch)`, `DIFF_ARCHS`):
  - "agil"   → UNet pequeña a 32×32, base 64, sin atención. Rápida (lo de siempre).
  - "nitido" → resolución NATIVA 64×64 (sin reescalado 32→64 que emborrona), UNet mayor
               (base 96, multiplicadores (1,2,2,4) → 64→32→16→8) con bloques de
               auto-atención multi-cabeza en las resoluciones 16 y 8. Es la de calidad.

RENDIMIENTO (MPS/CPU): "agil" trabaja a 32×32×3 con pocos canales y `T=200` (muestreo
interactivo con DDIM y pocos pasos). "nitido" es bastante más pesada (más canales, más
resolución y atención): mejor calidad a cambio de muestreo más lento. El schedule y la
parametrización del reverse son las estándar de DDPM y son AGNÓSTICOS a la resolución.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

IMG_CH = 3
IMG_SIZE = 32          # resolución por defecto (= variante "agil"); cada modelo expone su .img_size
BASE = 64              # canales base por defecto (= variante "agil")
TIME_DIM = 128         # dimensión del embedding temporal
ATTN_HEADS = 4         # nº de cabezas de la auto-atención (variante "nitido")

# ---- variantes disponibles (en orden de presentación en la UI) ----------------------
DIFF_ARCHS = ("agil", "nitido")


@dataclass(frozen=True)
class ArchSpec:
    """Configuración de una variante de la UNet de difusión.

    - `img_size`  : resolución nativa a la que trabaja el modelo (datos y muestreo).
    - `base`      : canales de la primera resolución; se multiplican por `ch_mults`.
    - `ch_mults`  : multiplicadores de canales por nivel (cada nivel baja ×2 la resolución).
    - `attn_res`  : resoluciones (en píxeles) donde insertar auto-atención.
    """

    img_size: int
    base: int
    ch_mults: tuple[int, ...]
    attn_res: tuple[int, ...]


# "agil": la UNet de SIEMPRE (in_conv→64 a 32×32; dos niveles 64→128→128, 32→16→8;
#   bottleneck 8×8 @128; sin atención). `ch_mults=(2,2)` reproduce exactamente esa red.
# "nitido": resolución nativa 64×64, base 96, (1,2,2,4) → 64→32→16→8, atención en 16 y 8.
ARCH_SPECS: dict[str, ArchSpec] = {
    "agil": ArchSpec(img_size=32, base=64, ch_mults=(2, 2), attn_res=()),
    "nitido": ArchSpec(img_size=64, base=96, ch_mults=(1, 2, 2, 4), attn_res=(16, 8)),
}

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
def _group_norm(ch: int) -> nn.GroupNorm:
    """GroupNorm con un nº de grupos seguro (divisor de `ch`, ≤8). Evita assert con bases
    como 96 (no divisible por 8 en algún nivel: 96 sí, pero los multiplicadores grandes dan
    canales como 384 que sí lo son; este helper es robusto para cualquier `base`)."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return nn.GroupNorm(g, ch)
    return nn.GroupNorm(1, ch)


class SelfAttention2d(nn.Module):
    """Auto-atención espacial multi-cabeza sobre el mapa de activaciones (GroupNorm + MHA).

    Aplana H*W en una secuencia de "tokens" de `ch` canales y deja que cada posición atienda
    a TODAS las demás (atención global), algo que las convoluciones —locales— no capturan.
    Es la pieza que más ayuda a la coherencia global de la cara en la variante "nitido"; se
    inserta solo en resoluciones bajas (16, 8) porque su coste crece con (H*W)².

    Residual: `x + Attn(Norm(x))`, con una proyección de salida inicializada a cero para que
    el bloque empiece como la identidad (entrenamiento estable).
    """

    def __init__(self, ch: int, heads: int = ATTN_HEADS) -> None:
        super().__init__()
        # nº de cabezas que divida `ch` (de `heads` hacia abajo)
        h = heads
        while h > 1 and ch % h != 0:
            h -= 1
        self.norm = _group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.heads = h
        self.head_dim = ch // h
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, hgt, wid = x.shape
        qkv = self.qkv(self.norm(x))                      # (B, 3C, H, W)
        q, k, v = qkv.chunk(3, dim=1)
        # (B, heads, head_dim, H*W) -> (B, heads, H*W, head_dim)
        q = q.view(b, self.heads, self.head_dim, hgt * wid).transpose(-1, -2)
        k = k.view(b, self.heads, self.head_dim, hgt * wid).transpose(-1, -2)
        v = v.view(b, self.heads, self.head_dim, hgt * wid).transpose(-1, -2)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        out = attn @ v                                    # (B, heads, H*W, head_dim)
        out = out.transpose(-1, -2).reshape(b, c, hgt, wid)
        return x + self.proj(out)


class ResBlock(nn.Module):
    """Bloque residual con inyección del embedding temporal (FiLM-additivo) y atención opcional."""

    def __init__(self, cin: int, cout: int, time_dim: int, attn: bool = False) -> None:
        super().__init__()
        self.norm1 = _group_norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, cout)
        self.norm2 = _group_norm(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attn = SelfAttention2d(cout) if attn else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return self.attn(h + self.skip(x))


class Down(nn.Module):
    """Bloque residual (con atención opcional) + downsample ×2 (stride-2 conv)."""

    def __init__(self, cin: int, cout: int, time_dim: int, attn: bool = False) -> None:
        super().__init__()
        self.block = ResBlock(cin, cout, time_dim, attn=attn)
        self.down = nn.Conv2d(cout, cout, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.block(x, t_emb)   # para la skip connection (antes de bajar resolución)
        return self.down(skip), skip


class Up(nn.Module):
    """Upsample ×2 + concat de la skip + bloque residual (con atención opcional).

    `in_ch`: canales de la entrada (del nivel inferior); `skip_ch`: canales de la skip que
    se concatena; `out_ch`: canales de salida del bloque.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int, attn: bool = False) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.block = ResBlock(in_ch + skip_ch, out_ch, time_dim, attn=attn)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class TimeUNet(nn.Module):
    """UNet PARAMETRIZABLE condicionada en `t` para predecir el ruido ε.

    `TimeUNet(img_size, base, ch_mults, attn_res)` construye un encoder-decoder simétrico:
    cada nivel de bajada divide la resolución entre 2 y multiplica los canales por el factor
    correspondiente de `ch_mults`; el bottleneck (dos ResBlock) opera en la resolución más
    baja; el decoder sube de vuelta concatenando las skip connections de cada nivel.

    Atención: se inserta auto-atención (`SelfAttention2d`) en TODO nivel cuya resolución esté
    en `attn_res`, tanto en el camino down como en el bottleneck y en el up correspondiente.

    Ejemplos (los de las variantes):
      - "agil"   : img_size=32, base=64, ch_mults=(1,2,2), attn_res=()      → 32→16→8.
      - "nitido" : img_size=64, base=96, ch_mults=(1,2,2,4), attn_res=(16,8) → 64→32→16→8.

    El modelo expone `.img_size` (lo usan el muestreo y el ruido inicial) y `.arch` (nombre).
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        base: int = BASE,
        ch_mults: tuple[int, ...] = (1, 2, 2),
        attn_res: tuple[int, ...] = (),
        arch: str = "agil",
    ) -> None:
        super().__init__()
        self.img_size = int(img_size)
        self.arch = str(arch)
        attn_set = set(int(r) for r in attn_res)

        # `in_conv` define un nivel "0" de `base` canales a resolución plena; cada nivel de
        # bajada produce `base*ch_mults[i]` canales y divide la resolución entre 2. El decoder
        # es simétrico y vuelve a `base` arriba. p.ej.:
        #   agil   : in_conv=64;  down → [128,128]            (32→16→8),   sin atención.
        #   nitido : in_conv=96;  down → [96,192,192,384]     (64→32→16→8), atención en 16 y 8.
        level_ch = [base * m for m in ch_mults]
        n_levels = len(level_ch)

        # MLP del embedding temporal compartido por todos los bloques
        self.time_mlp = nn.Sequential(
            nn.Linear(TIME_DIM, TIME_DIM),
            nn.SiLU(),
            nn.Linear(TIME_DIM, TIME_DIM),
        )
        self.in_conv = nn.Conv2d(IMG_CH, base, 3, padding=1)  # img_size×img_size, base

        # --- camino DOWN: el nivel i toma la salida anterior (base en i=0) y saca level_ch[i] ---
        self.downs = nn.ModuleList()
        res = self.img_size
        skip_ch: list[int] = []   # canales de la skip que produce cada nivel (= su salida)
        cin = base
        for i in range(n_levels):
            cout = level_ch[i]
            self.downs.append(Down(cin, cout, TIME_DIM, attn=res in attn_set))
            skip_ch.append(cout)
            cin = cout
            res //= 2  # tras bajar, la resolución del SIGUIENTE nivel es la mitad

        # --- bottleneck en la resolución más baja (res) con DOS ResBlock; atención si toca ---
        cbott = level_ch[-1]
        mid_attn = res in attn_set
        self.mid1 = ResBlock(cbott, cbott, TIME_DIM, attn=mid_attn)
        self.mid2 = ResBlock(cbott, cbott, TIME_DIM, attn=False)

        # --- camino UP: simétrico; cada nivel concatena su skip y vuelve a la anchura del
        # nivel superior (el último vuelve a `base`, la anchura de in_conv). ---
        self.ups = nn.ModuleList()
        cprev = cbott
        for i in reversed(range(n_levels)):
            res *= 2  # subimos: la atención se decide a la resolución de ESTE nivel
            sch = skip_ch[i]
            cout = level_ch[i - 1] if i > 0 else base
            self.ups.append(Up(cprev, sch, cout, TIME_DIM, attn=res in attn_set))
            cprev = cout

        self.out_norm = _group_norm(base)
        self.out_conv = nn.Conv2d(base, IMG_CH, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(t, TIME_DIM))
        h = self.in_conv(x)
        skips: list[torch.Tensor] = []
        for down in self.downs:
            h, s = down(h, t_emb)
            skips.append(s)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        for up in self.ups:
            h = up(h, skips.pop(), t_emb)
        return self.out_conv(F.silu(self.out_norm(h)))


def build_diffusion(arch: str = "agil") -> TimeUNet:
    """Construye la UNet de la variante pedida. `arch` desconocida → 'agil' (degrada bien).

    Devuelve una `TimeUNet` con `.img_size` y `.arch` ya fijados a los de la variante.
    """
    arch = arch if arch in ARCH_SPECS else "agil"
    spec = ARCH_SPECS[arch]
    return TimeUNet(
        img_size=spec.img_size,
        base=spec.base,
        ch_mults=spec.ch_mults,
        attn_res=spec.attn_res,
        arch=arch,
    )


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

    # CRÍTICO: el ruido inicial vive a la resolución NATIVA del modelo (cada variante la
    # suya), NO en la constante global IMG_SIZE. El schedule es agnóstico a la resolución.
    img_size = getattr(model, "img_size", IMG_SIZE)
    x = torch.randn(n, IMG_CH, img_size, img_size, device=device)
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
