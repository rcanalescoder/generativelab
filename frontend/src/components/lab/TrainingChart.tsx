import type { LossPoint } from '../../lib/api'
import { cn } from '../../lib/cn'

interface Props {
  points: LossPoint[]
  className?: string
  color?: string
}

/** Curva de pérdida (loss vs epoch). SVG responsive, sin dependencias. */
export function TrainingChart({ points, className, color = 'var(--blue)' }: Props) {
  const W = 600
  const H = 150
  const pad = 10

  if (points.length === 0) {
    return (
      <div
        className={cn('flex items-center justify-center text-[13px] text-[color:var(--muted)]', className)}
        style={{ height: H }}
      >
        Entrena para ver la curva de pérdida en vivo
      </div>
    )
  }

  const losses = points.map((p) => p.loss)
  const min = Math.min(...losses)
  const max = Math.max(...losses)
  const span = max - min || 1
  const n = points.length
  const X = (i: number) => pad + (i / Math.max(n - 1, 1)) * (W - 2 * pad)
  const Y = (v: number) => pad + (1 - (v - min) / span) * (H - 2 * pad)
  const line = points.map((p, i) => `${X(i).toFixed(1)},${Y(p.loss).toFixed(1)}`).join(' ')
  const area = `${pad},${H - pad} ${line} ${(W - pad).toFixed(1)},${H - pad}`
  const last = points[points.length - 1]

  return (
    <div className={cn('relative', className)}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="block h-[150px] w-full">
        <defs>
          <linearGradient id="lossfill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={color} stopOpacity="0.18" />
            <stop offset="1" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>
        <polygon points={area} fill="url(#lossfill)" />
        <polyline
          points={line}
          fill="none"
          stroke={color}
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        <circle cx={X(n - 1)} cy={Y(last.loss)} r={3.5} fill={color} />
      </svg>
      <div className="pointer-events-none absolute right-2 top-1 font-mono text-[12px] text-[color:var(--ink)]">
        loss {last.loss.toFixed(4)}
      </div>
      <div className="pointer-events-none absolute left-2 top-1 font-mono text-[11px] text-[color:var(--muted)]">
        max {max.toFixed(4)}
      </div>
    </div>
  )
}
