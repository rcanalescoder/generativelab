# Lista de Mejoras — v3 «caras de calidad»

> **Documento vivo.** Registra el ciclo agéntico de la v3: qué teníamos al principio (v2), qué se
> probó, por qué funciona cada mejora y cuánto mejoró según las métricas. Es la fuente para
> rehacer el PDF del proyecto. Cada experimento queda además registrado, máquina-legible, en
> `experiments/ledger.jsonl`, con su rejilla de muestras en `experiments/grids/`.

**Estado:** 🟡 en curso (sesión del 2026-06-12).

---

## 0. Resumen ejecutivo (se completa al cerrar el ciclo)

| Modelo | KID×1000 v2 (antes) | KID×1000 v3 (después) | Veredicto visual |
|---|---|---|---|
| VAE | — | — | — |
| GAN | — | — | — |
| Diffusion | — | — | — |

*KID (Kernel Inception Distance): distancia entre la distribución de caras generadas y la de
caras reales, medida sobre features de InceptionV3. **Menor = mejor.** 0 sería indistinguible
del dataset real.*

---

## 1. Cómo medimos la calidad (el arnés V3.0)

El problema número uno de la v2 no era ningún modelo: era que **el laboratorio no podía medir la
calidad de generación**. Las búsquedas de `app/experiments/` ordenaban por PSNR/SSIM de
*reconstrucción* — y un VAE colapsado reconstruye «aceptablemente» mientras genera manchas.
Optimizábamos a ciegas la métrica equivocada.

La v3 añade un protocolo fijo de evaluación (`backend/app/experiments/quality.py` + `runner.py`):

- **KID** (Kernel Inception Distance, Bińkowski et al. 2018): MMD² insesgada con kernel polinómico
  sobre features de InceptionV3 (2048-d). Se eligió sobre FID porque su estimador es **insesgado
  con pocas muestras** (FID con n<10k tiene sesgo notable) y es estable en MPS. Protocolo:
  2.048 muestras generadas vs 2.048 reales de un held-out fijo (seed 1234), 10 subconjuntos.
  Para diffusion se usan menos muestras (512) porque generar es caro; se compara siempre
  diffusion-contra-diffusion con el mismo n.
- **Diversidad**: distancia media por pares entre features de las muestras generadas, y su ratio
  frente a la diversidad del dataset real (ratio ≈ 1 ⇒ tan variado como el dataset; ≈ 0 ⇒ colapso:
  todas las caras iguales).
- **Rejilla 8×8 con semilla fija** (`experiments/grids/<run>.png`): mismas z/semillas en todos los
  runs de un modelo ⇒ comparación visual directa antes/después.
- **Ledger** (`experiments/ledger.jsonl` + `experiments/LEADERBOARD.md`): cada run guarda config
  completa, semilla, tiempos, métricas y rutas. Reproducibilidad ante todo.
- PSNR/SSIM se mantienen, pero **solo** como métrica de reconstrucción (AE/VAE), nunca como
  métrica de generación.

Dependencia nueva: `torchvision` (InceptionV3). El AE queda fuera del ciclo: no genera del prior,
y su variante U-Net ya reconstruye nítido — no era parte del problema.

---

## 2. Punto de partida (v2): qué teníamos y por qué salía mal

### 2.1 Diagnóstico del código

1. **VAE — colapso del posterior por un desequilibrio de escala (el hallazgo principal).**
   La pérdida era `recon + β·KL` con la reconstrucción **promediada** sobre los 12.288 valores de
   cada imagen (`nn.MSELoss`, media por píxel) pero la KL **sumada** sobre las 128 dimensiones del
   latente (`vae_service.py`). Para pesar lo que pesa en el ELBO real, con esa parametrización β
   tendría que valer ~0,0001; con β=1,00 la KL pesaba **miles de veces de más**. Resultado: al
   modelo le salía más barato aplastar la KL a 0 → el posterior q(z|x) se pegaba al prior → z no
   llevaba información → el decoder pintaba siempre la «cara media» del dataset. Por eso las
   8 muestras de la pestaña VAE eran manchas marrones casi idénticas. El slider de la UI
   (paso mínimo 0,01) ni siquiera podía expresar el valor que lo compensaría.

2. **GAN — DCGAN «de libro» (2016) sin los estabilizadores modernos.** lr única para G y D, sin
   EMA del generador, sin label smoothing, sin augmentación, sin spectral norm; el demo entrenaba
   con 15.000 de las 43.102 imágenes y 25 epochs. El propio código admitía «la calidad será tosca».

3. **Diffusion — base técnica buena, presupuesto de entrenamiento corto.** EMA ✓, schedule cosine ✓,
   DDIM ✓, atención en la variante «nitido» ✓. Pero el último entrenamiento real fue 60 epochs con
   18.000 imágenes (`backend/_train_nitido.py`), y no había ningún checkpoint en el repo (la
   carpeta `checkpoints/` estaba vacía: la app arrancaba sin demo).

4. **Transversal — sin métrica de calidad de generación** (ver §1).

### 2.2 Baselines medidos (config demo v2, reproducidos fielmente)

Antes de tocar nada se reprodujo el «antes» con la config demo exacta de la v2 y se midió con
el protocolo nuevo. El entorno es un Mac Apple Silicon (MPS), torch 2.12.

| Run | Config v2 | KID×1000 ↓ | Diversidad (ratio) | Qué se ve en el grid |
|---|---|---|---|---|
| `vae-v2-baseline-basico` | full · 12 epochs · β=1 (sin reescalar) | **608,5** | **0,145** | 64 manchas marrones casi idénticas (la «cara media») |
| `gan-v2-baseline-basico` | 25 epochs · 15k imgs · lr única 2e-4 | **199,5** | 1,054 | caras anime reconocibles pero toscas: manchas, deformaciones, zonas lavadas |
| `diffusion-v2-baseline-agil` | *(pendiente de medir)* | — | — | — |

Dos lecturas importantes del baseline VAE:

- La pérdida final de entrenamiento era «buena» (0,075) y el PSNR de reconstrucción aceptable
  (12,1 dB): **las métricas de la v2 no veían el problema**. El KID (608) y la diversidad (0,145)
  sí lo ven: es el peor resultado posible, generación colapsada.
- En el log de entrenamiento se observa la firma del colapso: `kl 0.0076` al terminar — el
  término KL aplastado a ~0, tal y como predecía el diagnóstico (§2.1.1).

---

## 3. Mejoras del VAE (V3.1)

### M-VAE-1 · Equilibrar reconstrucción y KL («el bug de escala») + warmup de β

**Qué había.** `loss = recon + β·KL` con recon *promediada* sobre los 12.288 valores de la imagen
y KL *sumada* sobre las 128 dims del latente. Con β=1, la KL pesaba miles de veces más que en el
ELBO real → el optimizador la aplastaba a 0 → colapso del posterior → «cara media» siempre.

**Qué se cambió** (`backend/app/services/vae_service.py`):
1. La KL se divide por el nº de píxeles (`PIXELS = 12.288`) antes de entrar en la pérdida. Así
   recon y KL están en la misma escala («por píxel») y **β=1 vuelve a significar ELBO
   equilibrado**; el slider de β recupera su sentido de β-VAE (β>1 = más regular, β<1 = más nítido).
2. **Warmup de β**: durante el primer 30 % de las epochs, β sube linealmente de 0 a su valor.
   El decoder aprende primero a reconstruir; la regularización entra después. Es la receta
   estándar contra el colapso temprano (KL annealing).

**Por qué funciona.** El ELBO con decoder gaussiano implica una relación fija entre el término de
reconstrucción (suma sobre píxeles) y la KL (suma sobre dims). Si se promedia uno y se suma el
otro, β deja de medir lo que el usuario cree que mide. Reescalar restituye el equilibrio teórico.

**Efecto medido** *(pendiente: screens en curso — mismo presupuesto que el baseline, 12 epochs full)*

---

## 4. Mejoras del GAN (V3.2)

*(pendiente)*

---

## 5. Mejoras de Diffusion (V3.3)

*(pendiente)*

---

## 6. Checkpoints adoptados y reproducción

*(pendiente)*

---

## 7. Cronología de experimentos

*(pendiente: tabla generada desde `experiments/ledger.jsonl` al cierre)*
