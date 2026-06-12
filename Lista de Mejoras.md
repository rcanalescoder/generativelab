# Lista de Mejoras — v3 «caras de calidad»

> **Documento vivo.** Registra el ciclo agéntico de la v3: qué teníamos al principio (v2), qué se
> probó, por qué funciona cada mejora y cuánto mejoró según las métricas. Es la fuente para
> rehacer el PDF del proyecto. Cada experimento queda además registrado, máquina-legible, en
> `experiments/ledger.jsonl`, con su rejilla de muestras en `experiments/grids/`.

**Estado:** ✅ ciclo completado (sesión del 2026-06-12, ~2,5 h de cómputo MPS, 17 runs registrados).

---

## 0. Resumen ejecutivo

| Modelo | KID×1000 v2 (antes) | KID×1000 v3 (después) | Mejora | Veredicto visual |
|---|---|---|---|---|
| VAE | 608,5 | **121,3** | **−80 %** | de 64 manchas marrones idénticas a caras suaves, variadas y reconocibles |
| GAN | 199,5 | **37,5** | **−81 %** | caras nítidas y coloridas, ojos detallados; artefactos solo puntuales |
| Diffusion | 175,1 | **19,4** | **−89 %** | el mejor del laboratorio: detalle fino, alto contraste, máxima variedad |

*KID (Kernel Inception Distance): distancia entre la distribución de caras generadas y la de
caras reales, medida sobre features de InceptionV3. **Menor = mejor.** 0 sería indistinguible
del dataset real. La diversidad pasó de 0,145 (VAE colapsado) a ≈1,0 en los tres modelos.*

**Montaje antes/después listo para el PDF:** `experiments/grids/v3-antes-despues.png`
(y los 6 grids individuales 8×8, con semilla fija, en `experiments/grids/`).

Las tres palancas que explican el salto, en una línea cada una:
1. **VAE:** un bug de escala (KL sumada vs reconstrucción promediada) colapsaba el posterior — se
   reescaló la KL por píxel, con warmup de β y una guarda numérica en logσ².
2. **GAN:** le faltaban los estabilizadores modernos (EMA del generador, label smoothing,
   DiffAugment) y su presupuesto real de datos/epochs.
3. **Diffusion:** la base ya era buena; solo necesitaba entrenamiento largo (dataset completo,
   warmup de lr) con guardado reanudable para poder dárselo.
4. *(transversal)* Nada de esto era optimizable sin **medir la generación**: el KID + los grids
   de semilla fija convirtieron «mejorar las caras» en un experimento con números.

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
| `diffusion-v2-baseline-agil` | agil 32×32 · 30 epochs · 20k imgs | **175,1** | 1,130 | caras correctas pero blandas (la resolución de trabajo 32×32 reescalada a 64 emborrona) |

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

**Efecto medido** — mismo presupuesto exacto que el baseline (12 epochs, dataset completo),
solo cambia la pérdida:

| Run | β | KID×1000 ↓ | Diversidad (ratio) | PSNR recon (dB) |
|---|---|---|---|---|
| v2 baseline | 1,0 (sin reescalar) | 608,5 | 0,145 (colapso) | 12,10 |
| v3 M1 | 1,0 | 255,1 | 1,078 | 17,30 |
| v3 M1 | **0,5** | **241,5** | 1,097 | **17,90** |
| v3 M1 | 2,0 | 292,2 | 1,062 | 16,55 |

El mismo modelo, los mismos datos y el mismo tiempo de entrenamiento pasan de 64 manchas
idénticas a 64 caras anime distintas (suaves, como corresponde a un VAE). KID −60 %,
diversidad ×7,5 y +5,8 dB de reconstrucción **gratis**: solo era la escala de la pérdida.

### M-VAE-2 · Calibrado de β (β-VAE con el slider ya en su sitio)

Con la KL ya por píxel, β vuelve a ser interpretable y el barrido muestra el trade-off de
libro: β bajo → más nitidez y mejor KID a este presupuesto; β alto → latente más regular pero
muestras más borrosas. β=0,5 queda como valor de calidad; β=1,0 como valor pedagógico (ELBO).

A igual presupuesto corto (12 epochs), la variante «grande» (base 64 + resblocks) **no** supera
a la básica (KID 254 vs 255) y cuesta 5,6× más tiempo: a presupuestos cortos la capacidad no es
el cuello de botella. Se reevalúa con entrenamiento largo.

### M-VAE-3 · Estabilidad numérica del posterior (clamp de logσ²)

Hallazgo del propio ciclo: al lanzar el entrenamiento largo de la variante grande, la KL explotó
en la epoch 1 (≈4·10¹⁷). Causa: con el warmup, β es aún pequeño al principio y nada ancla
`logσ²`; en la arquitectura grande (más profunda, con BatchNorm) la cabeza `fc_logvar` puede
producir valores enormes y `exp(logσ²)` se desborda, envenenando el estado del optimizador.
Arreglo estándar: `logσ².clamp(-8, 8)` en `encode()` (`models/vae.py`) — inocuo en el rango de
trabajo normal (±6) y elimina la inestabilidad de raíz. Lección de libro: *el warmup de KL
necesita una guarda numérica en la varianza*.

**Entrenamiento final (40 epochs, dataset completo, β=0,5):**

| Run | Arch | KID×1000 ↓ | Diversidad | PSNR (dB) | Tiempo |
|---|---|---|---|---|---|
| v3-final-basico | basico | 195,1 | 1,086 | 18,45 | 107 s |
| v3-final-grande | grande | **121,3** | 0,976 | **19,09** | 598 s |

Dos lecciones: (a) el VAE básico final (KID 195) ya supera al **GAN** del baseline v2 (KID 200);
(b) la capacidad extra de la variante grande no aporta con 12 epochs pero **sí con 40**
(254→121): primero arregla la pérdida, luego escala el cómputo. Caras con ojos definidos,
brillos y peinados variados — suaves al estilo VAE, pero indiscutiblemente caras.

**Resultado V3.1: KID 608 → 121 (−80 %), diversidad 0,145 → 0,98.** Checkpoints adoptados como
demos de la app: `vae_basico_demo.pt` (rápido) y `vae_grande_demo.pt` (calidad).

---

## 4. Mejoras del GAN (V3.2)

El DCGAN de la v2 era correcto pero «desnudo» (receta original de 2016). La v3 añade los
cuatro estabilizadores estándar que más mejoran un DCGAN, todos como hiperparámetros nuevos de
`GANHyperParams` (activados por defecto, desactivables):

### M-GAN-1 · EMA del generador (`ema: true`)
Se muestrea con la **media móvil exponencial** de los pesos de G (decay 0,999), no con los pesos
«crudos» que oscilan con cada minibatch. El generador EMA promedia las últimas miles de
versiones de G → caras más limpias y coherentes gratis. (Se reutiliza la clase `EMA` que ya
existía en `models/diffusion.py` — la práctica que más mejora DDPM, aplicada también al GAN.)

### M-GAN-2 · TTUR: lrs separadas (`lr_g: 1e-4`, `lr_d: 2e-4`)
*Two Time-scale Update Rule*: el discriminador aprende algo más rápido que el generador. Un D
ligeramente «por delante» da gradientes más informativos y el juego oscila menos. (Además
alinea la implementación con la spec §4.3, que pedía `lr_G`/`lr_D`.)

### M-GAN-3 · Label smoothing unilateral (`label_smooth: 0.9`)
Al entrenar D, las imágenes reales valen 0,9 en vez de 1,0. Evita que D se vuelva sobreconfiado
y deje a G sin gradiente útil (un D «perfecto» mata el aprendizaje de G).

### M-GAN-4 · DiffAugment (`diffaug: true`)
Augmentación **diferenciable** (color + traslación + cutout, Zhao et al. 2020) aplicada a TODO
lo que ve D — reales y falsas, también en el paso de G. Como ambas se transforman igual, el
equilibrio del juego no cambia, pero D ya no puede memorizar imágenes concretas del dataset →
menos sobreajuste de D, entrenamiento estable y mejor calidad con datasets de decenas de miles
de imágenes. Implementada en `models/gan.py` (`diff_augment`).

**Efecto medido — primera sorpresa del ciclo (y una lección de libro).** A presupuesto corto
(25 epochs × 15k = 2.950 pasos, idéntico al baseline), los estabilizadores **pierden**:

| Run (mismo presupuesto que el baseline) | KID×1000 ↓ |
|---|---|
| v2 baseline (sin estabilizadores) | **199,5** |
| v3 stack completo (EMA+TTUR+smooth+aug) | 223,7 |
| v3 solo EMA | 351,1 |
| v3 EMA+TTUR+smooth (sin aug) | 414,2 |

¿Por qué? Los estabilizadores son **apuestas a largo plazo**: la EMA con decay 0,999 promedia
~1.000 pasos (un tercio de todo el run corto → llega «con retraso»); TTUR baja la lr de G a la
mitad (G avanza menos en el mismo tiempo); DiffAugment le pone el trabajo más difícil a D. Todos
frenan la convergencia temprana a cambio de estabilidad tardía. Conclusión metodológica para el
PDF: **un screen corto no sirve para evaluar mejoras de estabilidad — hay que medirlas al
presupuesto real.**

**A presupuesto completo (40 epochs × dataset completo = 13.500 pasos) la foto se invierte:**

| Run (40 epochs, 43.102 imgs) | KID×1000 ↓ | Diversidad |
|---|---|---|
| Receta v2 (control) | 70,9 | 0,923 |
| Stack v3 con TTUR (lr_G 1e-4 / lr_D 2e-4) | 56,5 | 0,889 |
| **Stack v3 sin TTUR (lr 2e-4 / 2e-4)** | **52,2** | 0,895 |

Lecturas: (a) solo dar al GAN su presupuesto real (dataset completo + 40 epochs) ya baja el KID
de 199→71; (b) el stack v3 añade un **−26 % adicional** (71→52); (c) TTUR no compensa aquí — la
lr reducida de G sigue frenando incluso a 13.500 pasos, así que el default v3 queda en lrs
iguales (2e-4) y TTUR como opción explorable en la UI (sliders `lr_G`/`lr_D` nuevos, spec §4.3).
Visualmente: caras nítidas con ojos definidos y gran variedad de estilos; artefactos puntuales.

**Final adoptado (60 epochs, receta ganadora):**

| Run | KID×1000 ↓ | Diversidad | Tiempo |
|---|---|---|---|
| full60-v3-sinttur (EMA+smooth+DiffAugment, lr 2e-4/2e-4) | **37,5** | 0,907 | 21,5 min |

**Resultado V3.2: KID 199,5 → 37,5 (−81 %).** Caras nítidas y coloridas con ojos detallados;
artefactos solo puntuales. Checkpoint adoptado como demo: `gan_basico_demo.pt`. Defaults de la
app actualizados a la receta ganadora (lrs iguales 2e-4; EMA, smoothing y DiffAugment activos;
TTUR disponible vía los nuevos sliders `lr_G`/`lr_D`).

Trayectoria completa del GAN: 199,5 (v2) → 70,9 (su presupuesto real) → 52,2 (stack v3, 40 ep)
→ **37,5** (stack v3, 60 ep).

---

## 5. Mejoras de Diffusion (V3.3)

La base técnica de la v2 ya era buena (EMA ✓, schedule cosine ✓, muestreo DDIM ✓, atención en
la variante «nitido» ✓): aquí el cuello de botella no era un bug sino el **presupuesto de
entrenamiento** (60 epochs × 18k imágenes en el último run real) y que no había ningún
checkpoint versionado/regenerable de calidad.

### M-DIFF-1 · Entrenador largo y reanudable (`app/experiments/train_diffusion_v3.py`)
- **Dataset completo** (43.102 imágenes) en vez de 18k.
- **Warmup de lr** (500 pasos): estabiliza el arranque de la UNet.
- **Guardado doble periódico**: cada N epochs escribe (a) el checkpoint demo con pesos EMA
  que carga la app y (b) el estado completo (modelo crudo + optimizador + EMA + epoch) en
  `experiments/ckpts/` para **reanudar** con `--resume` sin perder progreso. Entrenar mucho
  tiempo deja de ser arriesgado: se puede cortar y seguir otra noche.
- Rejilla de progreso por guardado (`experiments/grids/diffusion-nitido-v3-progress-*.png`).

### M-DIFF-2 · Evaluación honesta del muestreo
La pérdida de denoising (MSE de ε) **no mide calidad de muestra**; ahora cada checkpoint se
evalúa con el mismo KID del resto de modelos (512 muestras, DDIM 80 pasos — generar con
difusión es caro, así que diffusion se compara siempre contra diffusion con el mismo n).

**Efecto medido** (eval con 512 muestras y DDIM 80 pasos en ambos):

| Run | Config | KID×1000 ↓ | Diversidad |
|---|---|---|---|
| v2 baseline (agil) | 32×32 · 30 epochs · 20k imgs | 175,1 | 1,130 |
| **v3 nitido (28 epochs)** | 64×64 nativo · atención · 43.102 imgs · EMA · warmup | **19,4** | 0,984 |

**Resultado V3.3: KID 175 → 19,4 (−89 %), el mejor modelo del laboratorio** — como cabía esperar
de un DDPM bien entrenado. 28 epochs × 128 s ≈ 60 min de entrenamiento en MPS; la pérdida seguía
bajando al cortar (0,0616 y descendiendo), así que hay margen extra con más noches
(`--resume` continúa donde se quedó). Checkpoint demo: `diffusion_nitido_demo.pt`.

---

## 6. Checkpoints adoptados y reproducción

Los checkpoints **no se versionan** en git (pesan decenas/cientos de MB y se regeneran en
minutos); lo que se versiona es el código + este documento + el ledger. Comandos exactos
(desde `backend/`, venv activado, `export PYTORCH_ENABLE_MPS_FALLBACK=1`):

| Demo | Comando | Tiempo (M-series) |
|---|---|---|
| `vae_basico_demo.pt` (KID 195) | `python -m app.experiments.runner --model vae --tag demo --mode full --hp '{"epochs":40,"beta":0.5,"arch":"basico"}' --save-ckpt app/checkpoints/vae_basico_demo.pt` | ~2 min |
| `vae_grande_demo.pt` (KID 121) | ídem con `"arch":"grande"` y `--save-ckpt app/checkpoints/vae_grande_demo.pt` | ~10 min |
| `gan_basico_demo.pt` (KID 37,5) | `python -m app.experiments.runner --model gan --tag demo --mode full --hp '{"epochs":60}' --save-ckpt app/checkpoints/gan_basico_demo.pt` | ~22 min |
| `diffusion_nitido_demo.pt` | `python -m app.experiments.train_diffusion_v3 --epochs 28` (reanudable con `--resume`) | ~1-2 h |

Cualquier checkpoint se evalúa con el protocolo oficial sin reentrenar:
`python -m app.experiments.runner --model <m> --load-demo <arch> --tag eval --eval-n 2048`
(diffusion: `--eval-n 512 --steps 80`). Los resultados van solos al ledger y al leaderboard.

---

## 7. Cronología de experimentos (2026-06-12, generada desde `experiments/ledger.jsonl`)

| # | Hora | Modelo | Run | KID×1000 ↓ | Div. | Train (s) |
|---|---|---|---|---|---|---|
| 1 | 20:27 | vae | v2-baseline-basico | 608,54 | 0,145 | 34 |
| 2 | 20:30 | gan | v2-baseline-basico | 199,53 | 1,054 | 140 |
| 3 | 20:32 | vae | v3-kl-beta1-basico | 255,06 | 1,078 | 33 |
| 4 | 20:32 | vae | v3-kl-beta05-basico | 241,47 | 1,097 | 33 |
| 5 | 20:33 | vae | v3-kl-beta2-basico | 292,18 | 1,062 | 33 |
| 6 | 20:36 | vae | v3-kl-beta1-grande | 254,22 | 1,023 | 183 |
| 7 | 20:39 | vae | v3-final-basico | 195,14 | 1,086 | 107 |
| 8 | 20:51 | vae | **v3-final-grande** | **121,30** | 0,976 | 598 |
| 9 | 20:55 | gan | v3-stack-completo (screen) | 223,71 | 0,911 | 189 |
| 10 | 20:57 | gan | v3-sin-diffaug (screen) | 414,16 | 1,087 | 136 |
| 11 | 21:00 | gan | v3-solo-ema (screen) | 351,13 | 1,056 | 136 |
| 12 | 21:11 | gan | full-v2-control | 70,88 | 0,923 | 619 |
| 13 | 21:25 | gan | full-v3-ttur | 56,48 | 0,889 | 859 |
| 14 | 21:40 | gan | full-v3-sinttur | 52,17 | 0,895 | 860 |
| 15 | 22:02 | gan | **full60-v3-sinttur** | **37,49** | 0,907 | 1292 |
| 16 | 22:09 | diffusion | v2-baseline-agil | 175,14 | 1,130 | 359 |
| 17 | 23:10 | diffusion | **v3-nitido-ep28** | **19,40** | 0,984 | ~3600 |

Además quedó en el ledger la historia completa de cada run (config, semillas, métricas y ruta de
su grid). Incidencias dignas de mención: (a) la explosión numérica de la variante grande del VAE
(→ M-VAE-3); (b) los screens cortos del GAN dando el veredicto **opuesto** al presupuesto real
(→ lección metodológica de §4).
