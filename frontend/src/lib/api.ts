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

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init)
  if (!res.ok) {
    throw new Error(`API ${path} → ${res.status} ${res.statusText}`)
  }
  return res.json() as Promise<T>
}

export function getHealth(): Promise<Health> {
  return getJSON<Health>('/health')
}
