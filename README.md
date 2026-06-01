# Generative Models Visual Lab

Laboratorio visual **local** para *entender* modelos generativos (Autoencoder, VAE, GAN,
Diffusion) sobre un dataset de caras anime 64×64. Herramienta **educativa y de demostración**,
con acabado premium, pensada para tocar parámetros, ver qué ocurre y hacer capturas.

> La especificación completa y la hoja de ruta están en [`CLAUDE.md`](CLAUDE.md) y
> [`GUIA-FASES.md`](GUIA-FASES.md). Este README explica cómo arrancar el proyecto.

## Estado por fases

| Fase | Contenido | Estado |
|---|---|---|
| **0** | Andamiaje + sistema de diseño + `/api/health` | ✅ hecho |
| **1** | Autoencoder (dataset, entrenamiento SSE, mapa latente, vecinos, interpolación) | ✅ hecho |
| 2 | Pantallas "i" + modo captura | ⏳ |
| 3 | VAE · 4 GAN · 5 Diffusion · 6 Comparador | ⏳ |

## Requisitos

- **macOS con Apple Silicon** (cómputo en MPS; *fallback* a CPU). Nunca CUDA.
- **Node ≥ 20** y **npm** (probado con Node 25 / npm 11).
- **Python ≥ 3.11** (probado con 3.13).

## Estructura

```
generative-models-visual-lab/
├── CLAUDE.md / GUIA-FASES.md   # especificación y guía por fases
├── design/mockup.html          # referencia visual aprobada (tab Autoencoder)
├── frontend/                   # Vite + React + TS + Tailwind v4
└── backend/                    # FastAPI + PyTorch (venv propio)
```

## Backend (FastAPI + PyTorch)

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Arrancar (puerto 8000). El script exporta PYTORCH_ENABLE_MPS_FALLBACK=1.
./run.sh
# …o a mano:
PYTORCH_ENABLE_MPS_FALLBACK=1 uvicorn app.main:app --reload --port 8000
```

Comprobación rápida:

```bash
curl http://localhost:8000/api/health
# {"status":"ok","device":"mps","device_name":"Apple Silicon GPU (MPS)", ...}
```

### Dataset y checkpoint demo (Fase 1)

La primera vez hay que preparar los datos (descarga el `data.zip` de `huggan/anime-faces`,
~462 MB, y construye un cache local `backend/data/anime-faces/faces_64.npy`):

```bash
# desde backend/ con el venv activado
python -m app.data.dataset            # descarga + cachea el dataset (solo la 1ª vez)
python -m app.services.ae_service     # entrena el checkpoint demo del AE (~3 min en MPS)
```

Hecho esto, al arrancar el backend carga el checkpoint y el tab Autoencoder es usable al
instante. Si no quieres esperar, también puedes pulsar **Entrenar** (modo *Rápido*) en la UI:
genera un modelo utilizable en unos segundos.

> Nota: el `data.zip` publicado contiene **43.102** imágenes (la ficha del dataset menciona
> 21.551). La app usa y muestra el recuento real.

## Frontend (Vite + React + TS + Tailwind)

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

En desarrollo, Vite hace de **proxy** de `/api/*` → `http://localhost:8000`, así que basta con
tener el backend levantado para que la pill "MPS activo" del header se ponga en verde.

Scripts disponibles (`frontend/`):

| Script | Qué hace |
|---|---|
| `npm run dev` | Servidor de desarrollo con HMR |
| `npm run build` | Type-check (`tsc -b`) + build de producción |
| `npm run preview` | Sirve el build de producción |
| `npm run lint` | ESLint |

## Notas

- **Un solo sistema de componentes propio** en `frontend/src/components/ui` (sin shadcn).
  Los tokens de diseño viven en `frontend/src/styles/tokens.css`.
- El **dataset** (`huggan/anime-faces`) y los **checkpoints demo** se añaden en la Fase 1
  (se descargan y cachean en local la primera vez).
