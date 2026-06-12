import type { ControlHint, HintKind } from '../../content/types'

/** Texto del chip superior del tooltip, por categoría de control. */
export const HINT_KIND_LABEL: Record<HintKind, string> = {
  retrain: 'cambia el modelo · requiere reentrenar',
  live: 'interactivo · efecto inmediato',
  action: 'acción',
}

/** Cuerpo del tooltip (chip de categoría + texto). Lo reutilizan Hint y Button. */
export function HintPop({ hint }: { hint: ControlHint }) {
  return (
    <span className="gm-hint-pop" role="tooltip">
      {hint.kind && <span className={`gm-hint-tag is-${hint.kind}`}>{HINT_KIND_LABEL[hint.kind]}</span>}
      {hint.text}
    </span>
  )
}

interface HintProps {
  /** Ayuda a mostrar. Si es undefined no se pinta nada (los JSON pueden ir por detrás). */
  hint?: ControlHint
}

/** Icono ⓘ pequeño con tooltip al pasar el ratón (o al enfocar con teclado).
 *  Se coloca junto a la etiqueta de un control para explicar qué hace, si requiere
 *  reentrenar y qué efecto esperar. La ayuda vive en `content/<modelo>.json` → `hints`. */
export function Hint({ hint }: HintProps) {
  if (!hint) return null
  return (
    <span className="gm-hint">
      <button type="button" className="gm-hint-i" aria-label={`Ayuda: ${hint.text}`} data-chrome="true">
        i
      </button>
      <HintPop hint={hint} />
    </span>
  )
}
