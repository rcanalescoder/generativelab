import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { cn } from '../../lib/cn'

type Variant = 'default' | 'primary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  /** Icono a la izquierda del texto. */
  icon?: ReactNode
  /** Ocupa todo el ancho disponible. */
  full?: boolean
}

export function Button({
  variant = 'default',
  icon,
  full,
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
    </button>
  )
}
