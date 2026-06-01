interface SparklineProps {
  data: number[]
  width?: number
  height?: number
  color?: string
  className?: string
}

/** Mini-curva sin ejes, para el StatusLine y resúmenes compactos. */
export function Sparkline({
  data,
  width = 64,
  height = 18,
  color = 'var(--green)',
  className,
}: SparklineProps) {
  if (data.length < 2) {
    return <svg width={width} height={height} className={className} aria-hidden="true" />
  }
  const min = Math.min(...data)
  const max = Math.max(...data)
  const span = max - min || 1
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width
      const y = height - ((v - min) / span) * height
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')

  return (
    <svg width={width} height={height} className={className} fill="none" aria-hidden="true">
      <polyline
        points={points}
        stroke={color}
        strokeWidth={1.6}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
