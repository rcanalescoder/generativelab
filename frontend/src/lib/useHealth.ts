import { useEffect, useState } from 'react'
import { getHealth, type Health } from './api'

interface UseHealth {
  health: Health | null
  error: string | null
  loading: boolean
}

/** Sondea /api/health una vez al montar. Sin polling (CLAUDE.md §7). */
export function useHealth(): UseHealth {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    getHealth()
      .then((h) => {
        if (!alive) return
        setHealth(h)
        setError(null)
      })
      .catch((e: unknown) => {
        if (!alive) return
        setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
  }, [])

  return { health, error, loading }
}
