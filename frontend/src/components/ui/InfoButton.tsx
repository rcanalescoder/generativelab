import { InfoIcon } from '../../lib/icons'

interface InfoButtonProps {
  /** Abre la pantalla "i". En Fase 0 es un placeholder; se cablea en Fase 2. */
  onClick?: () => void
  label?: string
}

/** Botón redondo "i" que vive en la cabecera de cada tarjeta. */
export function InfoButton({ onClick, label = 'Más información' }: InfoButtonProps) {
  return (
    <button type="button" className="gm-info" onClick={onClick} aria-label={label} title={label}>
      <InfoIcon />
    </button>
  )
}
