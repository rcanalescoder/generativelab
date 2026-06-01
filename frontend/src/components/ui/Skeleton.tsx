import type { CSSProperties } from 'react'
import { cn } from '../../lib/cn'

interface SkeletonProps {
  className?: string
  style?: CSSProperties
}

/** Bloque con shimmer para estados de carga. */
export function Skeleton({ className, style }: SkeletonProps) {
  return <div className={cn('gm-skeleton', className)} style={style} aria-hidden="true" />
}
