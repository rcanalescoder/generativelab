import type { ReactNode } from 'react'
import { cn } from '../../lib/cn'

type TagVariant = 'warn' | 'live' | 'success'

interface TagProps {
  /** warn = "requiere reentrenar", live = "interactivo", success. */
  variant: TagVariant
  children: ReactNode
  className?: string
}

/** Pill de estado. Convención §3.5: ámbar = requiere reentrenar, verde = interactivo. */
export function Tag({ variant, children, className }: TagProps) {
  return <span className={cn('gm-tag', `is-${variant}`, className)}>{children}</span>
}
