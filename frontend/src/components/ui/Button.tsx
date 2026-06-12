import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/cn'
import type { ControlHint } from '../../content/types'
import { HintPop } from './Hint'

type Variant = 'default' | 'primary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  /** Icono a la izquierda del texto. */
  icon?: ReactNode
  /** Ocupa todo el ancho disponible. */
  full?: boolean
  /** Ayuda contextual: tooltip al pasar el ratón (qué hace el botón y qué esperar). */
  hint?: ControlHint
}

export function Button({
  variant = 'default',
  icon,
  full,
  hint,
  className,
  children,
  type = 'button',
  ...rest
}: ButtonProps) {
  return (
    <button
      type={type}
      className={cn(
        'gm-btn',
        variant === 'primary' && 'is-primary',
        variant === 'ghost' && 'is-ghost',
        full && 'w-full justify-center',
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
      {hint && <HintPop hint={hint} />}
    </button>
  )
}
