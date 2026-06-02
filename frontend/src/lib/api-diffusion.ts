/*
 * Cliente de API del modelo Diffusion (DDPM). Reutiliza los helpers comunes de ./api
 * (getJSON/postJSON/streamSSE/API_BASE). Los tipos se mantienen en sync con los modelos
 * Pydantic del backend (CLAUDE.md §5). Un modelo de difusión genera desde ruido puro; no
 * tiene encoder, así que no hay reconstrucción ni mapa latente aquí.
 */

import { getJSON, postJSON, streamSSE, API_BASE, type Device, type LossPoint } from './api'

export interface DiffusionHyperParams {
  learning_rate: number
  epochs: number
  batch_size: number
  timesteps: number
}

export interface DiffusionStatus {
  model: string
  trained: boolean
  training: boolean
  seed: number
  device: Device
  timesteps: number
  hyperparams: DiffusionHyperParams
  num_params: number
  loss_history: LossPoint[]
}

/** Un fotograma de la trayectoria de difusión (de ruido puro a cara). */
export interface TrajectoryFrame {
  /** Posición en la secuencia de muestreo (0 = ruido puro). */
  position: number
  /** Número total de posiciones de la trayectoria. */
  total: number
  /** Paso `t` del schedule en esa posición (T = ruido puro, 0 = imagen limpia). */
  t: number
  image: string
}

export type DiffusionTrainEvent =
  | { type: 'start'; epochs: number; n: number; mode: string; seed: number; hyperparams: DiffusionHyperParams }
  | { type: 'step'; epoch: number; step: number; loss: number }
  | { type: 'epoch'; epoch: number; epochs: number; step: number; loss: number; preview: string[] }
  | {
      type: 'done'
      epochs: number
      epochs_run?: number
      stopped_early?: boolean
      loss: number | null
      loss_history: LossPoint[]
    }
  | { type: 'cancelled'; epoch: number; step: number }
  | { type: 'error'; message: string }

export interface DiffusionTrainRequest {
  mode: 'full' | 'quick'
  seed: number
  hyperparams: Partial<DiffusionHyperParams>
  early_stop?: boolean
}

// ---------- estado ----------
export const getDiffusionStatus = () => getJSON<DiffusionStatus>('/diffusion/status')

// ---------- entrenamiento (SSE) ----------
/** Entrena por SSE el denoising. Llama onEvent por cada evento `data:`. */
export const trainDiffusion = (
  body: DiffusionTrainRequest,
  onEvent: (ev: DiffusionTrainEvent) => void,
  signal?: AbortSignal,
) => streamSSE<DiffusionTrainEvent>('/diffusion/train', body, onEvent, signal)

export const cancelDiffusionTraining = () =>
  fetch(`${API_BASE}/diffusion/train/cancel`, { method: 'POST' })

// ---------- generación ----------
/** Muestrea `n` imágenes desde ruido puro. `steps` = nº de pasos de muestreo (submuestreo de T). */
export const generateDiffusion = (n: number, seed: number, steps: number) =>
  postJSON<{ images: string[] }>('/diffusion/generate', { n, seed, steps }).then((r) => r.images)

/** Genera UNA imagen y devuelve `snapshots` instantáneas de la trayectoria (ruido → cara). */
export const getTrajectory = (seed: number, steps: number, snapshots: number) =>
  postJSON<{ frames: TrajectoryFrame[] }>('/diffusion/trajectory', { seed, steps, snapshots }).then(
    (r) => r.frames,
  )
