import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'
import { InfoButton } from './InfoButton'

type RiseOrder = 1 | 2 | 3 | 4 | 5

interface CardProps {
  children: ReactNode
  className?: string
  /** Orden de aparición escalonada al cargar (1–5). */
  rise?: RiseOrder
}

/** Tarjeta base: fondo blanco, borde, sombra y radio del sistema. */
export function Card({ children, className, rise }: CardProps) {
  return (
    <section
      className={cn('gm-card', rise && 'gm-card-rise', rise && `gm-rise-${rise}`, className)}
    >
      {children}
    </section>
  )
}

interface CardHeaderProps {
  title: ReactNode
  /** Si se pasa, muestra el botón "i" a la derecha. */
  onInfo?: () => void
  infoLabel?: string
  /** Acciones extra a la izquierda del botón "i" (dropdowns, iconos…). */
  actions?: ReactNode
}

/** Cabecera de tarjeta: título a la izquierda, acciones + botón "i" a la derecha. */
export function CardHeader({ title, onInfo, infoLabel, actions }: CardHeaderProps) {
  return (
    <div className="gm-ch">
      <h2>{title}</h2>
      <div className="gm-ch-actions">
        {actions}
        {onInfo && <InfoButton onClick={onInfo} label={infoLabel} />}
      </div>
    </div>
  )
}
