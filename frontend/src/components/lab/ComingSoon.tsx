import type { ReactNode } from 'react'
import { Card } from '../ui'
import { SparklesIcon } from '../../lib/icons'

interface ComingSoonProps {
  title: string
  description: string
  /** Etiqueta de fase, p. ej. "Fase 1". */
  phase?: string
  icon?: ReactNode
}

/** Placeholder premium para lengüetas aún no implementadas (Fase 0). */
export function ComingSoon({ title, description, phase, icon }: ComingSoonProps) {
  return (
    <Card rise={1}>
      <div className="gm-soon">
        <div className="badge">{icon ?? <SparklesIcon />}</div>
        <h3>{title}</h3>
        <p>{description}</p>
        {phase && (
          <span className="gm-tag is-warn" style={{ marginTop: 4 }}>
            {phase}
          </span>
        )}
      </div>
    </Card>
  )
}
