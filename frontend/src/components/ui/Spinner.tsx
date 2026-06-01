import { cn } from '../../lib/cn'

/** Spinner circular del sistema. */
export function Spinner({ className }: { className?: string }) {
  return <div className={cn('gm-spinner', className)} role="status" aria-label="Cargando" />
}
