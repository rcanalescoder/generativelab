/*
 * Cliente de API del VAE. Reutiliza los helpers genéricos y los tipos compartidos de `./api`
 * (getJSON, postJSON, streamSSE y los tipos de proyección/vecinos/interpolación), y añade lo
 * propio del VAE: el hiperparámetro β, la curva doble (recon vs KL) y la generación por muestreo
 * del prior. Los tipos se mantienen en sync con los modelos Pydantic del backend (CLAUDE.md §5).
 */

import {
  getJSON,
  postJSON,
  streamSSE,
  type AEMetrics,
  type Device,
  type GridEvent,
  type GridRequest,
  type InterpResult,
  type Neighbor,
  type NeighborsResult,
  type Projection,
  type ProjectionPoint,
  type ReconItem,
  type Sample,
} from './api'

// Re-exportamos los tipos reutilizados para que el tab importe todo desde un único módulo.
export type {
  AEMetrics,
  GridEvent,
  GridRequest,
  InterpResult,
  Neighbor,
  NeighborsResult,
  Projection,
  ProjectionPoint,
  ReconItem,
  Sample,
}

/** Variantes de arquitectura del VAE (espejo de `AEArch`, pero sin "unet"). */
export type VAEArch = 'basico' | 'grande'

export interface VAEHyperParams {
  latent_dim: number
  learning_rate: number
  epochs: number
  loss: 'mse' | 'l1'
  beta: number
  batch_size: number
  arch: VAEArch
}

/** Punto de la curva de entrenamiento: pérdida total + sus dos componentes (recon y KL). */
export interface VAELossPoint {
  epoch: number
  loss: number
  recon: number
  kl: number
}

export interface VAEStatus {
  model: string
  trained: boolean
  outdated: boolean
  training: boolean
  seed: number
  device: Device
  arch: VAEArch
  archs: VAEArch[]
  hyperparams: VAEHyperParams
  num_params: number
  loss_history: VAELossPoint[]
}

export type VAETrainEvent =
  | { type: 'start'; epochs: number; n: number; mode: string; seed: number; hyperparams: VAEHyperParams }
  | { type: 'step'; epoch: number; step: number; loss: number; recon_loss: number; kl_loss: number }
  | {
      type: 'epoch'
      epoch: number
      epochs: number
      step: number
      loss: number
      recon_loss: number
      kl_loss: number
      preview: string[]
    }
  | {
      type: 'done'
      epochs: number
      epochs_run?: number
      stopped_early?: boolean
      loss: number | null
      loss_history: VAELossPoint[]
    }
  | { type: 'cancelled'; epoch: number; step: number }
  | { type: 'error'; message: string }

export interface VAETrainRequest {
  mode: 'full' | 'quick'
  seed: number
  hyperparams: Partial<VAEHyperParams>
  early_stop?: boolean
}

// ---------- estado / hiperparámetros ----------
export const getVAEStatus = () => getJSON<VAEStatus>('/vae/status')
export const setVAEHyperparams = (patch: Partial<VAEHyperParams>) =>
  postJSON<VAEStatus>('/vae/hyperparams', patch)
/** Cambia de variante de arquitectura cargando su checkpoint demo al instante (si existe).
 *  `loaded=false` indica que esa variante aún no tiene demo entrenado. */
export const setVAEArch = (arch: VAEArch) =>
  postJSON<VAEStatus & { loaded: boolean }>('/vae/arch', { arch })
/** Métricas de reconstrucción (PSNR/SSIM/MSE) sobre un held-out fijo, con ejemplos en b64.
 *  El payload es compatible con `AEMetrics` (mismo esquema en el backend). */
export const getVAEMetrics = (n = 256) => getJSON<AEMetrics>(`/vae/metrics?n=${n}`)

// ---------- reconstrucción / generación ----------
export const reconstructVAE = (ids: number[], noise = 0) =>
  postJSON<{ items: ReconItem[] }>('/vae/reconstruct', { ids, noise }).then((r) => r.items)

/** Muestrea n caras del prior N(0,I). Rasgo clave del VAE frente al AE. */
export const generateVAE = (n: number, seed = 42) =>
  postJSON<{ images: string[] }>('/vae/generate', { n, seed }).then((r) => r.images)

// ---------- latente ----------
export const getVAEProjection = (method: 'pca' | 'umap' = 'pca') =>
  getJSON<Projection>(`/vae/projection?method=${method}`)
export const getVAENeighbors = (query_id: number, k: number) =>
  postJSON<NeighborsResult>('/vae/neighbors', { query_id, k })
export const computeVAEClusters = (n_clusters: number) =>
  postJSON<Projection>('/vae/cluster', { n_clusters })
export const interpolateVAE = (a_id: number, b_id: number, steps: number) =>
  postJSON<InterpResult>('/vae/interpolate', { a_id, b_id, steps })

// ---------- entrenamiento (SSE) ----------
export const cancelVAETraining = () => fetch('/api/vae/train/cancel', { method: 'POST' })

/** Entrena por SSE. Llama onEvent por cada evento. */
export const trainVAE = (
  body: VAETrainRequest,
  onEvent: (ev: VAETrainEvent) => void,
  signal?: AbortSignal,
) => streamSSE<VAETrainEvent>('/vae/train', body, onEvent, signal)

// ---------- búsqueda en rejilla (SSE) — reutiliza los tipos del AE ----------
export const gridSearchVAE = (
  body: GridRequest,
  onEvent: (ev: GridEvent) => void,
  signal?: AbortSignal,
) => streamSSE<GridEvent>('/vae/gridsearch', body, onEvent, signal)

export const cancelVAEGridSearch = () => fetch('/api/vae/train/cancel', { method: 'POST' })
