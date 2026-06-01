import type { CSSProperties } from 'react'
import { cn } from '../../lib/cn'

interface SliderProps {
  /** Etiqueta a la izquierda. Si se omite, no se pinta la fila nombre/valor. */
  label?: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  /** Formatea el valor mostrado en la caja mono (p. ej. (v) => (v/100).toFixed(2)). */
  format?: (value: number) => string | number
  disabled?: boolean
  className?: string
}

/** Slider con track relleno en azul hasta el valor y caja mono con el valor actual. */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format = (v) => v,
  disabled,
  className,
}: SliderProps) {
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0
  const fillStyle = { '--p': `${pct.toFixed(1)}%` } as CSSProperties

  return (
    <div className={cn('gm-ctrl', className)}>
      {label !== undefined && (
        <div className="row">
          <span className="name">{label}</span>
          <span className="gm-val">{format(value)}</span>
        </div>
      )}
      <input
        type="range"
        className="gm-range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        style={fillStyle}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </div>
  )
}
