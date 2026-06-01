# Cómo construirlo con Claude Code — guía por fases

Esta guía acompaña a `CLAUDE.md`. La idea: avanzar **una fase cada vez**, revisando entre medias,
sin perder de vista el destino (que ya está descrito completo en `CLAUDE.md`).

---

## Antes de empezar (una sola vez)

1. Crea la carpeta del proyecto, por ejemplo `generative-models-visual-lab/`.
2. Mete dentro:
   - `CLAUDE.md` en la **raíz** (Claude Code lo lee solo al arrancar en esa carpeta).
   - La maqueta en `design/mockup.html`.
3. (Opcional pero recomendable) `git init` ya, para que cada fase quede en su commit.
4. Abre Claude Code **dentro de esa carpeta**.
5. Mensaje inicial sugerido para que se sitúe:

> Lee `CLAUDE.md` y `design/mockup.html`. No escribas código todavía. Cuéntame con tus
> palabras qué vamos a construir, cómo está organizado y cuál es el plan de fases. Cuando lo
> tengas claro, dímelo y te lanzo la Fase 0.

Eso verifica que ha "entendido" el proyecto antes de tocar nada.

---

## Cómo usar cada prompt

- Pega el bloque **tal cual** cuando quieras lanzar esa fase.
- Todos asumen que `CLAUDE.md` está presente y manda.
- Cada fase termina pidiendo **commit + parar**. No pegues la siguiente hasta haber revisado y probado.
- Si algo no te convence, **díselo en lenguaje natural** antes de avanzar ("el mapa latente lo
  quiero más alto", "estos botones más juntos"…). Es más barato corregir dentro de la fase que después.
- Si cambiáis una decisión de fondo, pídele que **actualice `CLAUDE.md`** para que no se pierda.

---

## FASE 0 — Andamiaje y sistema de diseño

```
Vamos con la FASE 0. Lee CLAUDE.md entero y respétalo. En esta fase NO implementes nada de
ML ni lógica de modelos: solo el esqueleto y el sistema de diseño.

Haz:
1. Monorepo con /frontend y /backend según la estructura de CLAUDE.md §1.
2. Frontend con Vite + React + TypeScript + Tailwind. Extrae los tokens de diseño
   (paleta, tipografías Geist/Geist Mono, sombras, radios, espaciados) de design/mockup.html
   y déjalos centralizados en src/styles/tokens.css + la config de Tailwind.
3. Implementa los componentes base de CLAUDE.md §2 (Card+CardHeader+InfoButton, Button,
   Slider, SegmentedControl, Tag/Pill, Tabs, StatusLine, Sparkline, Thumb, skeletons).
   Que se vean idénticos a la maqueta.
4. Layout global y header iguales a la maqueta: logo, título "Generative Models Visual Lab",
   lengüetas Autoencoder/VAE/GAN/Diffusion/Comparador, pill "MPS activo", botones
   "Modo captura" y "Exportar PNG" (de momento sin función). Ancho completo, tope 1920px.
5. Las 5 lengüetas enrutadas, cada una con un placeholder "En construcción".
6. Backend FastAPI con /api/health que devuelva el device detectado (mps/cpu) y la versión de
   torch. CORS para el front. Detección de MPS y PYTORCH_ENABLE_MPS_FALLBACK=1. Arranque con uvicorn.
7. README con cómo arrancar front y back, y scripts npm.

Criterios de aceptación:
- npm run dev levanta el front y se ve el header + lengüetas como en la maqueta.
- /api/health responde con el device.
- Cambiar de lengüeta funciona; todas muestran placeholder.
- Cero dependencias de shadcn u otras librerías de componentes.

Cuando esté: commit "phase 0: scaffolding + design system" y PARA. Espera mi revisión.
```

---

## FASE 1 — Autoencoder (vertical slice completo)

```
Vamos con la FASE 1: el tab Autoencoder completo y real, replicando design/mockup.html.
Respeta CLAUDE.md (§4.1 para los paneles, §3 para las convenciones, §5 para la API).

Backend:
1. Carga del dataset huggan/anime-faces siguiendo CLAUDE.md §3.6 (descarga+cache local, con
   fallback a carpeta local; nada de streaming HF).
2. Modelo Autoencoder convolucional (CLAUDE.md §4.1) en backend/app/models/ae.py.
3. Entrenamiento por SSE (eventos epoch/step/loss + preview opcional), con modo "full" y "quick",
   guardado de checkpoint y carga de un checkpoint DEMO preentrenado para usar la app al instante.
4. Endpoints: status, train (SSE), reconstruct (+diff), embed, projection (PCA por defecto, UMAP
   diferido), neighbors (distancia en z), cluster (k-means), interpolate, upload.

Frontend (tab Autoencoder = maqueta, reutilizando componentes de §2):
5. Dataset visual, Flujo del modelo (pipeline), Controles (hiperparámetros + exploración +
   StatusLine + acciones), Mapa latente (LatentMap con clusters, leyenda, tooltip, selector
   PCA/UMAP), Resultados (vecinos + interpolación A→B). Subir imagen → reconstruir.
6. Implementa la semilla visible (§3.4) y el indicador "requiere reentrenar" vs "interactivo" (§3.5).
7. Curva de pérdida en vivo durante el entrenamiento (TrainingChart vía SSE) y botón de cancelar.
8. Estados de cargando/vacío/error en cada panel (§3.10).

NO toques aún las pantallas "i" ni el modo captura (son la Fase 2). Deja el botón i e/o capture
como placeholder si hace falta.

Criterios de aceptación:
- Con el checkpoint demo cargado, el tab se ve y funciona como la maqueta sin entrenar nada.
- "Entrenar" (modo quick) muestra la curva avanzando en vivo por SSE.
- Reconstruir, buscar similares, calcular clusters e interpolar funcionan sobre datos reales.
- Cambiar latent_dim marca el modelo como "requiere reentrenar".

Cuando esté: commit "phase 1: autoencoder vertical slice" y PARA. Espera mi revisión.
```

---

## FASE 2 — Pantallas "i" (gramática de tarjetas) + Modo captura

```
Vamos con la FASE 2: la infraestructura transversal de las pantallas "i" y el modo captura.
Respeta CLAUDE.md §3.8 y §3.9. Esto se reutilizará en TODAS las lengüetas.

Haz:
1. InfoPanel montado por "gramática de tarjetas": bloques ConceptCard, ParameterCard,
   DiagramBlock y EquationBlock. El contenido se lee de JSON por modelo en
   frontend/src/content/<model>.json. El panel es GRANDE y pensado para capturas.
2. El botón i de cada tarjeta abre el InfoPanel con el bloque/tema correspondiente.
3. Escribe el contenido JSON del Autoencoder (content/autoencoder.json) como primer ejemplo
   completo: qué es un autoencoder, qué hace cada hiperparámetro, el diagrama del pipeline y la
   noción de espacio latente. (Las fórmulas pueden renderizarse con KaTeX o similar.)
4. Modo captura: un CaptureFrame que encuadra la vista en formatos 16:9, 4:5, 9:16 y 1:1, con
   "Exportar PNG" a dimensiones fijas por formato (1920×1080, 1080×1350, 1080×1920, 1080×1080) y
   salida nítida a escala 2x (html-to-image o canvas). Cablea de verdad los botones "Modo captura"
   y "Exportar PNG" del header.

Criterios de aceptación:
- Pulsar la i en las tarjetas del Autoencoder abre un panel claro y bonito, alimentado por JSON.
- Añadir/editar texto del panel es solo tocar el JSON, sin tocar componentes.
- El modo captura cambia el encuadre y exporta un PNG nítido en cada formato.

Cuando esté: commit "phase 2: info panels + capture mode" y PARA. Espera mi revisión.
```

---

## FASE 3 — VAE

```
Vamos con la FASE 3: el tab VAE. Respeta CLAUDE.md §4.2 (paneles), §3 (convenciones) y §5 (API).
Reutiliza los componentes de §2 y los paneles de components/lab; difiere solo en datos/paneles nuevos.

Backend:
1. Modelo VAE (encoder → μ, logσ²; reparametrización; decoder) con pérdida recon + β·KL.
2. Entrenamiento por SSE emitiendo DOS series (recon y KL). Checkpoint demo. Modo quick/full.
3. Endpoints: status, train (SSE), reconstruct, embed, projection, neighbors, cluster, interpolate,
   y los propios del VAE: sample (del prior, con seed y truncation opcional) y traverse (recorrer
   una/dos dimensiones latentes).

Frontend (tab VAE):
4. Reusa el layout del AE y añade: muestreo del prior (SampleGrid), latent traversal (sliders por
   dimensión; si latent_dim==2, manifold 2D), curva doble recon/KL, vista prior vs posterior,
   interpolación y mapa latente.
5. Control de β con su pill "requiere reentrenar".
6. Contenido content/vae.json para el InfoPanel (reparametrización, término KL, qué hace β).

Criterios de aceptación:
- Con checkpoint demo, "Generar muestras" produce caras nuevas desde el prior.
- El latent traversal muestra cómo cambia la cara al mover una dimensión.
- La curva de entrenamiento muestra recon y KL por separado.

Cuando esté: commit "phase 3: VAE tab" y PARA. Espera mi revisión.
```

---

## FASE 4 — GAN

```
Vamos con la FASE 4: el tab GAN (DCGAN 64×64). Respeta CLAUDE.md §4.3, §3 y §5. Reutiliza componentes.

Backend:
1. Generator + Discriminator (DCGAN) con pérdidas adversariales.
2. Entrenamiento por SSE emitiendo G loss y D loss; durante el entrenamiento, guardar muestras de
   un z FIJO por epoch (para la rejilla evolutiva). Checkpoint demo. Modo quick/full.
3. Endpoints: status, train (SSE), sample (con truncation), fixed-grid?epoch=, interpolate,
   discriminator-scores (real vs fake).

Frontend (tab GAN):
4. Curvas G vs D (TrainingChart con dos series). Rejilla de z fijo con timeline/slider de epoch.
   Muestreo de z nuevo (SampleGrid). Paseo latente/interpolación (Filmstrip). Slider de truncation ψ.
   Histograma de confianza del discriminador (real vs fake). Métrica simple de diversidad/colapso (proxy).
5. Contenido content/gan.json para el InfoPanel (minimax, inestabilidad, trucos).

Nota: NO bloquees la fase por FID real. Si quieres una métrica, usa un proxy ligero o déjala marcada
como opcional.

Criterios de aceptación:
- Con checkpoint demo, se generan caras y se ve la rejilla de z fijo evolucionando por epoch.
- Las curvas G/D se actualizan en vivo durante el entrenamiento.
- El slider de truncation cambia visiblemente calidad/diversidad.

Cuando esté: commit "phase 4: GAN tab" y PARA. Espera mi revisión.
```

---

## FASE 5 — Diffusion

```
Vamos con la FASE 5: el tab Diffusion (DDPM pequeño). Respeta CLAUDE.md §4.4, §3 y §5. Reutiliza componentes.

Backend:
1. UNet pequeña condicionada en t, entrenada por predicción de ruido ε, con schedule linear/cosine.
2. Entrenamiento por SSE. Checkpoint demo. Modo quick/full.
3. Endpoints: status, train (SSE), forward?t= (imagen ruidosa en el paso t), sample (SSE o lista con
   pasos intermedios; sampler DDPM/DDIM y nº de pasos), schedule (β_t, α_t, ᾱ_t).

Frontend (tab Diffusion):
4. Forward: Filmstrip x0→xT con slider de t. Reverse: Filmstrip xT→x0 (denoising); generar desde
   ruido puro mostrando pasos intermedios (idealmente en streaming). Curvas del schedule. Vista
   ε predicho vs real. Control de pasos de muestreo y selector de sampler (DDIM para acelerar).
5. Contenido content/diffusion.json para el InfoPanel (q(x_t|x0), reverse, objetivo de entrenamiento).

Criterios de aceptación:
- El slider de t muestra el ruido aumentando en el forward.
- "Generar" produce una cara desde ruido mostrando la trayectoria de denoising.
- DDIM con pocos pasos hace el muestreo razonablemente rápido para interactuar.

Cuando esté: commit "phase 5: Diffusion tab" y PARA. Espera mi revisión.
```

---

## FASE 6 — Comparador + pulido final

```
Vamos con la FASE 6: el tab Comparador y el pulido final. Respeta CLAUDE.md §4.5 y §3.

Comparador:
1. Misma semilla → generación lado a lado de VAE/GAN/Diffusion (el AE muestra reconstrucción).
2. Misma imagen de entrada → reconstrucción AE vs VAE.
3. Tabla de métricas: nº de parámetros, tiempo de entrenamiento, pérdida final, error de
   reconstrucción (AE/VAE), notas cualitativas.
4. Radar comparando calidad de muestra, diversidad, estabilidad de entrenamiento, velocidad e
   interpretabilidad del latente.
5. Comparación de interpolación y tarjetas pros/contras por modelo.
6. Endpoints compare/sample y compare/metrics.

Pulido global:
7. Revisa estados de cargando/vacío/error en todas las lengüetas. Persistencia de checkpoints entre
   arranques. Animaciones coherentes. Repasa que el modo captura y las pantallas "i" funcionen en
   las cuatro lengüetas (con su JSON). README final completo (instalación, arranque, descarga del
   dataset, checkpoints demo, cómo entrenar).

Criterios de aceptación:
- El Comparador genera con la misma semilla en los modelos generativos y muestra la tabla y el radar.
- Las cuatro lengüetas tienen su pantalla "i" y funcionan en modo captura.
- README permite a alguien nuevo arrancar todo de cero.

Cuando esté: commit "phase 6: comparator + polish" y PARA.
```

---

## Consejos finales

- **Revisa de verdad entre fases.** Arranca la app, toca todo, haz capturas. Es el momento de
  decir "esto me gusta / esto no" mientras es barato cambiarlo.
- **Una fase = un commit.** Si una fase se hace grande, pídele que la parta en sub-commits, pero
  no mezcles fases.
- **El `CLAUDE.md` es vivo.** Cada vez que decidáis algo nuevo (un color, un endpoint, una
  decisión de arquitectura), pídele que lo anote ahí. Así la siguiente sesión arranca con todo.
- **Si se desvía**, recondúcelo recordándole la sección concreta: "esto incumple CLAUDE.md §3.8,
  la teoría va en JSON".
- **Empezar siempre por el demo checkpoint.** Que cada modelo sea usable sin entrenar te ahorra
  esperas y hace las demos inmediatas.
```
