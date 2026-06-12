/*
 * Cliente de API del GAN. Reutiliza los helpers de ./api (getJSON/postJSON/streamSSE).
 * Los tipos se mantienen en sync con los modelos Pydantic del backend (app/routers/gan.py).
 *
 * A diferencia del autoencoder, el GAN no reconstruye ni tiene mapa latente: genera caras
 * desde ruido z e interpola en ese espacio de ruido.
 */

import { getJSON, postJSON, streamSSE, type Device } from './api'

/** Variante de arquitectura del GAN (capacidad del generador/discriminador). */
export type GANArch = 'basico' | 'grande'

export interface GANHyperParams {
  z_dim: number
  learning_rate: number
  epochs: number
  batch_size: number
  arch: GANArch
  // v3: TTUR (lrs separadas para G y D; null → se usa learning_rate) + estabilizadores
  lr_g: number | null
  lr_d: number | null
  label_smooth: number
  ema: boolean
  diffaug: boolean
}

/** Punto de la curva doble: una pérdida para el generador y otra para el discriminador. */
export interface GANLossPoint {
  epoch: number
  g_loss: number
  d_loss: number
}

export interface GANStatus {
  model: string
  trained: boolean
  outdated: boolean
  training: boolean
  seed: number
  device: Device
  z_dim: number
  arch: GANArch
  archs: GANArch[]
  hyperparams: GANHyperParams
  num_params: number
  num_params_g: number
  num_params_d: number
  loss_history: GANLossPoint[]
}

export interface GANInterpFrame {
  alpha: number
  image: string
}

export interface GANInterpResult {
  steps: number
  seed: number
  frames: GANInterpFrame[]
}

/** Eventos SSE del entrenamiento del GAN (con g_loss/d_loss y preview de un z fijo). */
export type GANTrainEvent =
  | { type: 'start'; epochs: number; n: number; mode: string; seed: number; hyperparams: GANHyperParams }
  | { type: 'step'; epoch: number; step: number; g_loss: number; d_loss: number }
  | {
      type: 'epoch'
      epoch: number
      epochs: number
      step: number
      g_loss: number
      d_loss: number
      preview: string[]
    }
  | { type: 'done'; epochs: number; g_loss: number | null; d_loss: number | null; loss_history: GANLossPoint[] }
  | { type: 'cancelled'; epoch: number; step: number }
  | { type: 'error'; message: string }

export interface GANTrainRequest {
  mode: 'full' | 'quick'
  seed: number
  hyperparams: Partial<GANHyperParams>
}

// ---------- status / hiperparámetros ----------
export const getGANStatus = () => getJSON<GANStatus>('/gan/status')
export const setGANHyperparams = (patch: Partial<GANHyperParams>) =>
  postJSON<GANStatus>('/gan/hyperparams', patch)
/** Cambia de variante de arquitectura cargando su checkpoint demo al instante (si existe).
 *  `loaded=false` indica que esa variante aún no tiene demo entrenado. */
export const setGANArch = (arch: GANArch) =>
  postJSON<GANStatus & { loaded: boolean }>('/gan/arch', { arch })

// ---------- entrenamiento (SSE) ----------
/** Entrena el GAN por SSE. Llama onEvent por cada evento (incluye g_loss/d_loss/preview). */
export const trainGAN = (
  body: GANTrainRequest,
  onEvent: (ev: GANTrainEvent) => void,
  signal?: AbortSignal,
) => streamSSE<GANTrainEvent>('/gan/train', body, onEvent, signal)

export const cancelGANTraining = () =>
  fetch('/api/gan/train/cancel', { method: 'POST' })

// ---------- generación / interpolación ----------
export const generateGAN = (n: number, seed: number) =>
  postJSON<{ images: string[] }>('/gan/generate', { n, seed }).then((r) => r.images)

export const interpolateGAN = (steps: number, seed: number) =>
  postJSON<GANInterpResult>('/gan/interpolate', { steps, seed })
