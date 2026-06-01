import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { SparkIcon } from '../../lib/icons'

interface StatusLineProps {
  /** Contenido libre, p. ej. <>epoch <b>12/30</b> · loss <span className="mono">0.0184</span></>. */
  children: ReactNode
  /** Icono inicial; por defecto la chispa verde. */
  icon?: ReactNode
  className?: string
}

/** Línea de estado del entrenamiento: "epoch 12/30 · loss 0.0184". */
export function StatusLine({ children, icon, className }: StatusLineProps) {
  return (
    <div className={cn('gm-status', className)}>
      <span className="spark">{icon ?? <SparkIcon width={20} height={14} />}</span>
      <span>{children}</span>
    </div>
  )
}
