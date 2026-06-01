import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import type { Topic } from '../../content/types'
import { InfoPanel } from './InfoPanel'

interface InfoCtx {
  open: (topic: Topic) => void
  close: () => void
}

const Ctx = createContext<InfoCtx | null>(null)

/** Acceso al panel "i" desde cualquier parte (cabeceras de tarjeta, header…). */
export function useInfo(): InfoCtx {
  const c = useContext(Ctx)
  if (!c) throw new Error('useInfo debe usarse dentro de <InfoProvider>')
  return c
}

export function InfoProvider({ children }: { children: ReactNode }) {
  const [topic, setTopic] = useState<Topic | null>(null)
  const value = useMemo<InfoCtx>(() => ({ open: setTopic, close: () => setTopic(null) }), [])

  return (
    <Ctx.Provider value={value}>
      {children}
      {topic && <InfoPanel topic={topic} onClose={() => setTopic(null)} />}
    </Ctx.Provider>
  )
}
