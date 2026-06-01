/*
 * Cliente de API. En desarrollo Vite hace de proxy de /api → http://localhost:8000
 * (ver vite.config.ts), así que usamos rutas relativas. Los tipos se mantienen en
 * sync con los modelos Pydantic del backend (CLAUDE.md §5).
 */

export const API_BASE = '/api'

export type Device = 'mps' | 'cpu'

export interface Health {
  status: string
  device: Device
  device_name: string
  torch_version: string
  python_version: string
  mps_available: boolean
  mps_fallback: boolean
}

export interface DatasetInfo {
  name: string
  count: number
  image_shape: number[]
  license: string
}

export interface Sample {
  id: number
  image: string
}

export interface AEHyperParams {
  latent_dim: number
  learning_rate: number
  epochs: number
  loss: 'mse' | 'l1'
  batch_size: number
}

export interface LossPoint {
  epoch: number
  loss: number
}

export interface AEStatus {
  model: string
  trained: boolean
  outdated: boolean
  training: boolean
  seed: number
  device: Device
  hyperparams: AEHyperParams
  num_params: number
  loss_history: LossPoint[]
}

export interface ReconItem {
  id: number
  original: string
  reconstruction: string
  diff: string
}

export interface ProjectionPoint {
  id: number
  x: number
  y: number
  znorm: number
  cluster: number | null
}

export interface Projection {
  points: ProjectionPoint[]
  method: string
  n_clusters: number
}

export interface Neighbor {
  id: number
  distance: number
  image: string
}

export interface NeighborsResult {
  query: { id: number; image: string }
  neighbors: Neighbor[]
}

export interface InterpFrame {
  alpha: number
  image: string
}

export interface InterpResult {
  a_id: number
  b_id: number
  steps: number
  frames: InterpFrame[]
}

export type TrainEvent =
  | { type: 'start'; epochs: number; n: number; mode: string; seed: number; hyperparams: AEHyperParams }
  | { type: 'step'; epoch: number; step: number; loss: number }
  | { type: 'epoch'; epoch: number; epochs: number; step: number; loss: number; preview: string[] }
  | { type: 'done'; epochs: number; loss: number | null; loss_history: LossPoint[] }
  | { type: 'cancelled'; epoch: number; step: number }
  | { type: 'error'; message: string }

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body?.detail) detail = body.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

function postJSON<T>(path: string, body: unknown): Promise<T> {
  return getJSON<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

// ---------- comunes ----------
export const getHealth = () => getJSON<Health>('/health')
export const getDatasetInfo = () => getJSON<DatasetInfo>('/dataset/info')
export const getSamples = (n = 28, seed = 0) =>
  getJSON<{ samples: Sample[] }>(`/dataset/samples?n=${n}&seed=${seed}`).then((r) => r.samples)
export const getImage = (id: number) => getJSON<Sample>(`/dataset/image/${id}`)

// ---------- autoencoder ----------
export const getAEStatus = () => getJSON<AEStatus>('/autoencoder/status')
export const setAEHyperparams = (patch: Partial<AEHyperParams>) =>
  postJSON<AEStatus>('/autoencoder/hyperparams', patch)
export const reconstruct = (ids: number[], noise = 0) =>
  postJSON<{ items: ReconItem[] }>('/autoencoder/reconstruct', { ids, noise }).then((r) => r.items)
export const getProjection = (method: 'pca' | 'umap' = 'pca') =>
  getJSON<Projection>(`/autoencoder/projection?method=${method}`)
export const getNeighbors = (query_id: number, k: number) =>
  postJSON<NeighborsResult>('/autoencoder/neighbors', { query_id, k })
export const computeClusters = (n_clusters: number) =>
  postJSON<Projection>('/autoencoder/cluster', { n_clusters })
export const interpolate = (a_id: number, b_id: number, steps: number) =>
  postJSON<InterpResult>('/autoencoder/interpolate', { a_id, b_id, steps })

export async function uploadImage(file: File): Promise<ReconItem> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`upload → ${res.status}`)
  const data = (await res.json()) as Omit<ReconItem, 'id'>
  return { id: -1, ...data }
}

export const cancelTraining = () => fetch(`${API_BASE}/autoencoder/train/cancel`, { method: 'POST' })

export interface TrainRequest {
  mode: 'full' | 'quick'
  seed: number
  hyperparams: Partial<AEHyperParams>
}

/** Entrena por SSE sobre fetch (POST con body). Llama onEvent por cada evento. */
export async function trainAE(
  body: TrainRequest,
  onEvent: (ev: TrainEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_BASE}/autoencoder/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.body) throw new Error('sin stream de respuesta')
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const line = block.split('\n').find((l) => l.startsWith('data:'))
      if (line) onEvent(JSON.parse(line.slice(5).trim()) as TrainEvent)
    }
  }
}
