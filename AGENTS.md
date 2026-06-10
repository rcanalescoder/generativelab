# Generative Models Visual Lab — Especificación del proyecto

> Este archivo es la **fuente de verdad** del proyecto. Codex debe leerlo entero al
> empezar cualquier sesión y respetarlo. Si una instrucción puntual contradice este documento,
> pregunta antes de continuar.

---

## 0. Qué estamos construyendo y cómo trabajamos

**Qué es:** un laboratorio visual **local** para *entender* modelos generativos sobre un
dataset de caras anime de 64×64. Es una herramienta **educativa y de demostración**, con
acabado premium, pensada para hacer capturas de pantalla y explicar conceptos. No es un
producto SaaS ni va a desplegarse en la nube.

**Para quién:** un ingeniero de IA junior (máster de IA). Prioriza claridad pedagógica,
reproducibilidad y un código limpio y legible por encima de la sofisticación.

**Cómo trabajamos — regla de oro:**
- Construcción **incremental por fases**. Al terminar cada fase: hacer **commit**, dejar el
  estado funcionando y **PARAR**. No empezar la fase siguiente hasta que el humano la lance.
- Nada de "ya que estoy, adelanto la fase 3". Cada fase tiene un alcance cerrado.
- Si algo del diseño o del enfoque genera dudas, **proponer y preguntar**, no improvisar en silencio.

**Plataforma:** Mac con Apple Silicon. Cómputo en **MPS** con *fallback* a CPU. **Nunca CUDA.**
Exportar siempre `PYTORCH_ENABLE_MPS_FALLBACK=1` y detectar el device en arranque.

**Referencia visual:** `/design/mockup.html` es la maqueta del tab **Autoencoder** ya aprobada.
De ahí salen el "chrome" (header, lengüetas, tarjetas) y todos los tokens de diseño. El resto de
lengüetas deben sentirse hermanas de esa.

---

## 1. Stack y estructura

**Monorepo:**

```
generative-models-visual-lab/
├── AGENTS.md                  # este archivo
├── README.md
├── design/
│   └── mockup.html            # referencia visual del tab Autoencoder
├── frontend/                  # Vite + React + TS + Tailwind
│   ├── src/
│   │   ├── components/ui/      # sistema de componentes propio (§2)
│   │   ├── components/lab/     # piezas compartidas del laboratorio
│   │   ├── tabs/               # Autoencoder, VAE, GAN, Diffusion, Comparator
│   │   ├── lib/                # api client, hooks, utilidades (SSE, captura PNG…)
│   │   ├── content/            # JSON de las pantallas "i" por modelo (§3)
│   │   ├── styles/tokens.css   # CSS variables (paleta, sombras, radios)
│   │   └── App.tsx
│   └── ...
└── backend/                   # FastAPI + PyTorch
    ├── app/
    │   ├── main.py            # FastAPI, CORS, routers
    │   ├── device.py         # detección MPS/CPU
    │   ├── data/             # carga del dataset + fallback local
    │   ├── models/           # ae.py, vae.py, gan.py, diffusion.py
    │   ├── routers/          # un router por modelo + comunes
    │   ├── services/         # entrenamiento, proyección, clustering, sampling
    │   └── checkpoints/      # pesos guardados (incluye los de "demo")
    └── ...
```

**Decisiones ya tomadas (no reabrir sin avisar):**
- **Un único sistema de componentes propio** en `components/ui`. **No** usar shadcn ni otra
  librería de componentes. Tailwind sí, como utilidades.
- **Comunicación front↔back:** REST para todo lo puntual + **SSE** para el *streaming* de
  entrenamiento (progreso epoch a epoch). Nada de *polling*.
- TypeScript estricto en el front. Tipos de la API compartidos/derivados del backend (Pydantic).
- Gestor de paquetes front: el que prefieras (npm/pnpm), pero deja scripts claros en README.
- Backend en un **venv** propio; documenta el `pip install`. (No uses flags raros: es un Mac normal.)

---

## 2. Sistema de diseño (tokens y componentes)

Modo **claro premium**. Extrae y respeta exactamente lo que hay en `/design/mockup.html`.
Resumen de tokens (déjalos como CSS variables en `styles/tokens.css` y refléjalos en la config de Tailwind):

**Color**
- Acentos: `--blue:#2563EB`, `--cyan:#06B6D4`, `--violet:#8B5CF6`, `--pink:#EC4899`,
  `--green:#10B981`, `--amber:#F59E0B`.
- Suaves (fondos de chip/tarjeta): `blue-soft #EAF1FE`, `violet-soft #F1ECFC`,
  `pink-soft #FCE9F2`, `green-soft #E5F6F0`, `amber-soft #FCEFD7`.
- Neutros: fondo `#F5F6F9`, tarjeta `#FFFFFF`, tinta `#111827`, slate `#5B6472`,
  apagado `#8A93A2`, línea `#EDEFF4`, track de slider `#E7EAF1`.

**Tipografía**
- Cuerpo: **Geist**. Números, fórmulas y código: **Geist Mono** (con `font-feature-settings:"tnum"`).
- Cargar desde Google Fonts. Fallback a sans del sistema.

**Forma**
- Radios: tarjetas 18px, controles ~11px. Pills full-rounded.
- Sombra de tarjeta: `0 1px 3px rgba(17,24,39,.05), 0 10px 30px -12px rgba(17,24,39,.10)`.
- Layout a **ancho completo** (tope 1920px), padding lateral ~34px.
- Animaciones sutiles: aparición escalonada de tarjetas al cargar, *hover* leve, pulso del punto verde "MPS activo".

**Componentes base de `components/ui` (mínimos):**
`Card` (+ `CardHeader` con título e `InfoButton`), `Button` (variantes: default, primary, ghost),
`Slider` (track con relleno en color, thumb blanco con anillo, valor mono en una caja),
`SegmentedControl` (tipo MSE/L1), `Tag`/`Pill` (variantes warn="requiere reentrenar",
live="interactivo", success), `Tabs` (las lengüetas), `StatusLine` (línea "epoch 12/30 · loss 0.0184"
con mini-sparkline), `Sparkline`, `Thumb` (miniatura de imagen con borde), `Spinner`/skeletons.

**Componentes de `components/lab` (compartidos entre tabs):**
`DatasetCard` (galería de muestras), `LatentMap` (scatter 2D con clusters, leyenda, tooltip,
selector PCA/UMAP, descargar), `SampleGrid` (rejilla de imágenes generadas/reconstruidas),
`Filmstrip` (tira horizontal de fotogramas, p. ej. interpolación o difusión), `TrainingChart`
(curva(s) de pérdida en vivo), `ImageDiff` (mapa |x−x̂|), `InfoPanel` (la pantalla "i", §3),
`CaptureFrame` (encuadre del modo captura, §3).

> Importante: estos componentes se diseñan **una vez** y se reutilizan en todas las lengüetas.
> Un panel nuevo casi nunca es un componente nuevo; suele ser una composición de los anteriores.

---

## 3. Convenciones transversales (aplican a TODAS las lengüetas)

1. **Entrenamiento por SSE.** El endpoint de entrenar emite eventos `{epoch, step, ...losses, preview?}`
   en streaming. El front pinta la curva y el estado en vivo. Botón para cancelar.
2. **Modo demo con checkpoint preentrenado.** Cada modelo debe poder cargar pesos ya entrenados
   desde `backend/app/checkpoints/` para que la app sea usable y "capturable" al instante, sin
   esperar a entrenar. El entrenamiento en vivo es para *enseñar la dinámica*, no un requisito para ver resultados.
3. **Modo "quick" de entrenamiento.** Además del entrenamiento completo, ofrecer un modo rápido
   (pocas epochs / subconjunto) para demos en directo sin esperas largas.
4. **Semilla fija y visible.** Toda generación/proyección usa una seed mostrada en la UI y editable.
   Reproducibilidad ante todo.
5. **Indicador "requiere reentrenar".** Los hiperparámetros que cambian el modelo (p. ej. `latent_dim`)
   marcan el modelo como *desactualizado* (pill ámbar) hasta reentrenar. Los parámetros de exploración
   (ruido, k, clusters, pasos) son **interactivos** y no requieren reentrenar (pill verde).
6. **Dataset:** `huggan/anime-faces` (43.102 imágenes 64×64 RGB, CC0 — la ficha del dataset menciona 21.551, pero el `data.zip` publicado trae 43.102). **No** usar streaming del
   dataset viewer de HuggingFace (es inestable). Descargar el zip / usar `snapshot_download` una vez
   y cachear en local; si falla la red, *fallback* a una carpeta local de imágenes. La primera vez se
   descarga; después, todo local.
7. **Proyección 2D:** **PCA por defecto** (rápida, sin dependencias pesadas). UMAP **opcional y diferido**
   (cárgalo solo si el usuario lo pide; si no está instalado, degrada con elegancia a PCA y avísalo).
8. **Pantallas "i" = gramática de tarjetas.** El botón `i` de cada tarjeta abre un **panel grande,
   apto para capturas**, montado a partir de bloques reutilizables: `ConceptCard` (idea en lenguaje
   llano), `ParameterCard` (qué hace cada hiperparámetro y su rango), `DiagramBlock` (esquema),
   `EquationBlock` (fórmula con su lectura). **El contenido vive en JSON por modelo** en
   `frontend/src/content/<model>.json`. Así, añadir la teoría de un modelo nuevo es **escribir datos,
   no código**. La pantalla "i" no debe ser arquitectura nueva: es el mismo `InfoPanel` con otro JSON.
9. **Modo captura.** Botón global que mete la vista en un `CaptureFrame` con formatos **16:9, 4:5,
   9:16 y 1:1**, y **Exportar PNG** con **dimensiones fijas por formato** (p. ej. 1920×1080, 1080×1350,
   1080×1920, 1080×1080). Usar una librería tipo `html-to-image`/canvas. El PNG debe salir nítido (escala 2x).
10. **Estados.** Cada panel tiene estados coherentes de *cargando* (skeleton), *vacío* ("entrena o
    carga el checkpoint demo") y *error*. Nada de pantallas en blanco.
11. **Coste/feasibilidad.** Modelos **pequeños** a propósito (64×64). Lo lento en difusión es el
    muestreo: usa DDIM con pocos pasos para interactividad y apóyate en checkpoints demo.

---

## 4. Catálogo de experimentos por lengüeta  ← (las "directrices a priori")

Esta es la foto completa de lo que queremos poder hacer en cada pestaña. Se implementa por fases,
pero Codex debe conocerla entera desde el principio para no tomar decisiones que luego estorben
(estructura de datos del latente, contrato de API, componentes compartidos, etc.).

### 4.1 Autoencoder (determinista) — *módulo de referencia*
**Objetivo pedagógico:** compresión, cuello de botella, reconstrucción y estructura del espacio latente.
**Modelo:** encoder conv `64×64×3 → 4×4×256 → fc → z(latent_dim)`; decoder simétrico
(`ConvTranspose`) `z → 64×64×3`. Pérdida MSE o L1.
**Hiperparámetros (requieren reentrenar):** `latent_dim`, `learning_rate`, `epochs`, `loss (MSE/L1)`.
**Exploración (interactivo):** `ruido en z`, `k vecinos`, `n clusters`, `pasos de interpolación`.
**Paneles (= maqueta):**
- *Dataset visual*: galería de muestras + metadatos.
- *Flujo del modelo*: pipeline Original → Encoder → z (cuello de botella) → Decoder → Reconstruida → Diferencia |x−x̂|.
- *Controles*: los hiperparámetros + exploración + `StatusLine` + acciones (Entrenar, Reconstruir, Buscar similares, Interpolar, Calcular clusters).
- *Mapa latente*: proyección 2D (PCA/UMAP) de z, coloreada por cluster (k-means), con tooltips (id, cluster, ‖z‖) y marcadores miniatura.
- *Resultados*: vecinos más cercanos (distancia **en z**, no en 2D) e interpolación lineal A→B.
- *Extras*: subir imagen → reconstruir; rejilla original vs reconstruida; curva de pérdida en vivo.

### 4.2 VAE (Variational Autoencoder)
**Objetivo:** latente **probabilístico**, **generación** por muestreo del prior, continuidad/
disentanglement y el trade-off **reconstrucción vs KL**.
**Modelo:** encoder → `(μ, logσ²)`; reparametrización `z = μ + σ·ε`, `ε~N(0,I)`; decoder.
Pérdida = `recon (MSE/BCE) + β·KL`.
**Hiperparámetros:** `latent_dim`, `learning_rate`, `epochs`, **`β` (β-VAE)**, tipo de recon loss.
**Paneles nuevos respecto al AE (reutilizando componentes):**
- *Muestreo del prior*: `z~N(0,I)` → `SampleGrid` de caras generadas (botón "Generar muestras").
- *Latent traversal*: sliders para recorrer 1–2 dimensiones latentes con el resto fijas → ver qué
  controla cada eje (disentanglement). Si `latent_dim==2`, mostrar el **manifold 2D** (rejilla decodificada).
- *Curva doble*: recon loss **y** KL a lo largo del entrenamiento (`TrainingChart` con dos series).
- *Prior vs posterior*: histograma/scatter de los z de datos superpuestos a N(0,I).
- *Interpolación* (más suave que en AE) y *Mapa latente* (igual que AE, resaltando la gaussianidad).
**"i":** truco de reparametrización, por qué el término KL regulariza, qué hace β.

### 4.3 GAN (DCGAN 64×64)
**Objetivo:** entrenamiento **adversarial**, dinámica G/D, generación, colapso de modo y *truncation*.
**Modelo:** Generator (`z → imagen`, ConvTranspose) + Discriminator (`imagen → logit`).
Pérdidas adversariales (BCE o hinge).
**Hiperparámetros:** `dim z`, `lr_G`, `lr_D`, `epochs`, (`n_critic`), `batch`.
**Paneles:**
- *Curvas G loss vs D loss* (lo central: mostrar el tira y afloja).
- *Rejilla de z fijo que evoluciona por epoch*: se guardan muestras de un z fijo durante el
  entrenamiento; timeline/slider de epoch para ver cómo "nace" la cara (visualización GAN clásica).
- *Muestreo*: generar de z nuevo → `SampleGrid`.
- *Paseo latente / interpolación* en z (`Filmstrip`).
- *Truncation trick*: slider ψ que recorta z (calidad ↔ diversidad).
- *Confianza del discriminador*: histograma de `D(x)` sobre lote real vs lote falso.
- *Diversidad / colapso*: métrica simple de varianza entre muestras (proxy). **FID real es pesado:**
  marcarlo como opcional o usar un proxy ligero; no bloquear la fase por esto.
**"i":** el juego minimax, por qué es inestable, trucos de estabilización.

### 4.4 Diffusion (DDPM pequeño)
**Objetivo:** proceso **forward** (añadir ruido) y **reverse** (denoising), el *schedule* y el muestreo iterativo.
**Modelo:** UNet pequeña condicionada en `t`, entrenada por **predicción de ruido ε**. Schedule linear/cosine.
**Hiperparámetros:** `T (timesteps)`, `schedule`, `learning_rate`, `epochs`; **sampler (DDPM/DDIM)** y
`pasos de muestreo`.
**Paneles:**
- *Forward*: `Filmstrip` `x₀ → x_T` añadiendo ruido, con slider de `t`.
- *Reverse*: `Filmstrip` `x_T → x₀` (trayectoria de denoising); generar desde ruido puro mostrando
  pasos intermedios (idealmente en streaming).
- *Curvas del schedule*: `β_t`, `α_t`, `ᾱ_t`.
- *ε predicho vs real* (visual).
- *Control de pasos de muestreo*: DDIM para acelerar y poder interactuar.
**"i":** `q(x_t|x₀)`, la parametrización del reverse, el objetivo de entrenamiento.

### 4.5 Comparador
**Objetivo:** comparar los cuatro enfoques en igualdad de condiciones.
**Paneles:**
- *Misma semilla → generación lado a lado* de VAE / GAN / Diffusion (el AE no genera de prior: mostrar reconstrucción).
- *Misma imagen de entrada → reconstrucción* AE vs VAE (y donde aplique).
- *Tabla de métricas*: nº de parámetros, tiempo de entrenamiento, pérdida final, error de
  reconstrucción (AE/VAE), notas cualitativas de calidad/diversidad.
- *Radar* comparando: calidad de muestra, diversidad, estabilidad de entrenamiento, velocidad,
  interpretabilidad del latente.
- *Comparación de interpolación* (mismos extremos en cada modelo).
- *Tarjetas pros/contras* por modelo.

---

## 5. Contrato de API (orientativo)

Define los tipos en el backend (Pydantic) y mantenlos en sync con el front. Esquema base:

**Comunes**
- `GET  /api/health` → `{ device: "mps"|"cpu", torch_version, ... }`
- `GET  /api/dataset/info` → `{ name, count, image_shape, license }`
- `GET  /api/dataset/samples?n=` → miniaturas (URLs estáticas o base64)
- `GET  /api/{model}/status` → `{ trained, outdated, seed, hyperparams }`
- `POST /api/{model}/train` → **SSE**: eventos `{epoch, step, losses, preview?}`; acepta `{mode:"full"|"quick", hyperparams, seed}`
- `POST /api/{model}/train/cancel`

**Autoencoder / VAE**
- `POST /api/{model}/reconstruct` → imagen(es) reconstruidas (+ diff)
- `POST /api/{model}/embed` → z de un conjunto de muestras (para proyección)
- `GET  /api/{model}/projection?method=pca|umap` → `{points:[{x,y,id,cluster,znorm}], method}`
- `POST /api/{model}/neighbors` → vecinos por distancia en z `{query_id, k}`
- `POST /api/{model}/cluster` → etiquetas k-means `{n_clusters}`
- `POST /api/{model}/interpolate` → fotogramas A→B `{a_id, b_id, steps}`
- `POST /api/upload` → reconstruir imagen subida
- **VAE:** `POST /api/vae/sample` (del prior, `{n, seed, truncation?}`), `POST /api/vae/traverse` (`{dim, range, steps, base_z?}`)

**GAN**
- `POST /api/gan/sample` (`{n, seed, truncation}`)
- `GET  /api/gan/fixed-grid?epoch=` (muestras del z fijo en una epoch)
- `POST /api/gan/interpolate`
- `GET  /api/gan/discriminator-scores` (real vs fake)

**Diffusion**
- `GET  /api/diffusion/forward?t=` (imagen ruidosa en el paso t)
- `POST /api/diffusion/sample` → **SSE** o lista con pasos intermedios (`{steps, sampler, seed}`)
- `GET  /api/diffusion/schedule` (`β_t`, `α_t`, `ᾱ_t`)

**Comparador**
- `POST /api/compare/sample` (`{seed}`) → muestras de cada modelo
- `GET  /api/compare/metrics` → tabla agregada

> Convención: imágenes como PNG base64 o URLs estáticas servidas por FastAPI; latentes y curvas como JSON.

---

## 6. Hoja de ruta (resumen — el detalle y los prompts están en `GUIA-FASES.md`)

- **Fase 0 — Andamiaje + sistema de diseño.** Esqueleto, tokens, componentes base, 5 lengüetas
  placeholder, `/api/health`. *Sin ML.*
- **Fase 1 — Autoencoder (vertical slice).** El tab Autoencoder completo y real (= maqueta), con
  entrenamiento por SSE, checkpoint demo, proyección, vecinos, clusters, interpolación, subir imagen.
- **Fase 2 — Pantallas "i" + Modo captura.** `InfoPanel` por gramática de tarjetas (JSON), botón `i`
  grande para capturas, y modo captura con formatos y export PNG. Se demuestra sobre el AE.
- **Fase 3 — VAE.** Modelo probabilístico, muestreo del prior, latent traversal, curva recon/KL, β.
- **Fase 4 — GAN.** DCGAN, curvas G/D, rejilla de z fijo evolutiva, truncation, score del discriminador.
- **Fase 5 — Diffusion.** DDPM pequeño, forward/reverse filmstrips, schedule, sampler DDPM/DDIM.
- **Fase 6 — Comparador + pulido.** Comparativa lado a lado, métricas, radar; estados, persistencia, README final.

Cada fase termina con **commit + PARAR + esperar revisión**.

---

## 7. Cosas que NO hacer

- ❌ CUDA, o asumir GPU NVIDIA. Solo MPS/CPU.
- ❌ Streaming del dataset de HuggingFace (descargar + cachear local, con fallback).
- ❌ Mezclar shadcn u otra librería de componentes con el sistema propio.
- ❌ *Polling* para el progreso de entrenamiento (usar SSE).
- ❌ Convertir VAE/GAN/Diffusion en arquitecturas distintas del AE en el front: deben **reutilizar**
  los mismos componentes de `components/lab` y diferir solo en datos/paneles.
- ❌ Meter la teoría de los modelos en código: va en JSON (`content/<model>.json`).
- ❌ Bloquear una fase por métricas caras (FID): usar proxy u opcional.
- ❌ Adelantar fases o ampliar el alcance sin que el humano lo pida.
- ❌ Romper el contrato de API sin avisar y actualizar este documento.

---

## 8. Definición de "hecho" (por fase)

Una fase está hecha cuando: (a) cumple sus criterios de aceptación, (b) arranca sin errores
(`npm run dev` + `uvicorn`), (c) no rompe lengüetas ya existentes, (d) hay commit con mensaje claro,
y (e) el README refleja cualquier paso nuevo de instalación/arranque.
