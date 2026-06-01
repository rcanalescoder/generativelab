import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'

interface ThumbProps {
  /** URL de la imagen. Si falta, se renderiza `children` (p. ej. un SVG o skeleton). */
  src?: string
  alt?: string
  children?: ReactNode
  className?: string
}

/** Miniatura cuadrada con borde y radio del sistema. */
export function Thumb({ src, alt = '', children, className }: ThumbProps) {
  return <div className={cn('gm-thumb', className)}>{src ? <img src={src} alt={alt} /> : children}</div>
}
