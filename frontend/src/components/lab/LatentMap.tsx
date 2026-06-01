import { useRef, useState } from 'react'
import { Card, CardHeader, Skeleton } from '../ui'
import { ChevronDownIcon, DownloadIcon } from '../../lib/icons'
import { getImage, type Projection, type ProjectionPoint } from '../../lib/api'

const CLUSTER_COLORS = [
  '#3B82F6', '#8B5CF6', '#EC4899', '#10B981', '#F59E0B', '#06B6D4',
  '#F472B6', '#34D399', '#FBBF24', '#A78BFA', '#60A5FA', '#F87171',
]
const NEUTRAL = '#94A3B8'
const W = 660
const H = 300
const PAD = 18

const colorFor = (c: number | null) => (c == null ? NEUTRAL : CLUSTER_COLORS[c % CLUSTER_COLORS.length])

interface HoverState {
  point: ProjectionPoint
  px: number
  py: number
}

interface Props {
  projection: Projection | null
  method: 'pca' | 'umap'
  onMethodChange: (m: 'pca' | 'umap') => void
  loading?: boolean
  onInfo?: () => void
}

export function LatentMap({ projection, method, onMethodChange, loading, onInfo }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const cache = useRef<Map<number, string>>(new Map())
  const [hover, setHover] = useState<HoverState | null>(null)
  const [hoverImg, setHoverImg] = useState<string | undefined>(undefined)
  const [menuOpen, setMenuOpen] = useState(false)

  const points = projection?.points ?? []
  const nClusters = projection?.n_clusters ?? 0

  // extensión para escalar a la viewBox
  let minX = 0, maxX = 1, minY = 0, maxY = 1
  if (points.length) {
    const xs = points.map((p) => p.x)
    const ys = points.map((p) => p.y)
    minX = Math.min(...xs); maxX = Math.max(...xs)
    minY = Math.min(...ys); maxY = Math.max(...ys)
  }
  const sx = (W - 2 * PAD) / (maxX - minX || 1)
  const sy = (H - 2 * PAD) / (maxY - minY || 1)
  const toX = (x: number) => PAD + (x - minX) * sx
  const toY = (y: number) => PAD + (maxY - y) * sy // flip vertical

  const onEnter = (e: React.MouseEvent, p: ProjectionPoint) => {
    const rect = wrapRef.current?.getBoundingClientRect()
    if (!rect) return
    setHover({ point: p, px: e.clientX - rect.left, py: e.clientY - rect.top })
    const cached = cache.current.get(p.id)
    if (cached) {
      setHoverImg(cached)
    } else {
      setHoverImg(undefined)
      getImage(p.id)
        .then((s) => {
          cache.current.set(p.id, s.image)
          setHover((h) => (h && h.point.id === p.id ? h : h)) // mantener
          setHoverImg(s.image)
        })
        .catch(() => undefined)
    }
  }

  const download = () => {
    if (!projection) return
    const blob = new Blob([JSON.stringify(projection, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `latent-${method}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const legend =
    nClusters > 0
      ? Array.from({ length: nClusters }, (_, k) => ({ label: `Cluster ${k}`, color: colorFor(k) }))
      : [{ label: 'Muestras', color: NEUTRAL }]

  return (
    <Card className="mapa flex flex-col">
      <CardHeader
        title="Mapa latente"
        onInfo={onInfo}
        infoLabel="Qué es el espacio latente"
        actions={
          <>
            <button className="gm-iconbtn" onClick={download} title="Descargar proyección (JSON)" disabled={!projection}>
              <DownloadIcon />
            </button>
            <div className="relative">
              <button className="gm-dropdown" onClick={() => setMenuOpen((o) => !o)}>
                {method.toUpperCase()}
                <ChevronDownIcon />
              </button>
              {menuOpen && (
                <div className="absolute right-0 z-10 mt-1 w-28 overflow-hidden rounded-[10px] border border-[color:var(--line)] bg-white shadow-card">
                  {(['pca', 'umap'] as const).map((m) => (
                    <button
                      key={m}
                      className="block w-full px-3 py-2 text-left text-[13px] hover:bg-[color:var(--blue-soft)]"
                      onClick={() => {
                        onMethodChange(m)
                        setMenuOpen(false)
                      }}
                    >
                      {m.toUpperCase()}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </>
        }
      />
      <div className="gm-caption" style={{ margin: '-8px 0 12px' }}>
        Proyección 2D de z mediante {projection ? projection.method.toUpperCase() : 'PCA/UMAP'}
      </div>

      <div className="gm-latentwrap" ref={wrapRef} onMouseLeave={() => setHover(null)}>
        <div className="gm-legend">
          {legend.map((l) => (
            <div className="it" key={l.label}>
              <span className="sw" style={{ background: l.color }} />
              {l.label}
            </div>
          ))}
        </div>

        {loading || !projection ? (
          <Skeleton className="absolute inset-0 !rounded-[14px]" />
        ) : (
          <svg className="gm-scatter" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
            {points.map((p) => (
              <circle
                key={p.id}
                cx={toX(p.x)}
                cy={toY(p.y)}
                r={hover?.point.id === p.id ? 6 : 3}
                fill={colorFor(p.cluster)}
                opacity={hover && hover.point.id !== p.id ? 0.45 : 0.8}
                onMouseEnter={(e) => onEnter(e, p)}
              />
            ))}
          </svg>
        )}

        {hover && (
          <div
            className="gm-tooltip"
            style={{
              left: Math.min(hover.px + 12, (wrapRef.current?.clientWidth ?? W) - 190),
              top: Math.max(hover.py - 30, 0),
            }}
          >
            <div className="tp">
              {hoverImg ? <img src={hoverImg} alt="" /> : <Skeleton className="h-full w-full !rounded-[9px]" />}
            </div>
            <div className="tt">
              <div className="id">ID: {hover.point.id.toLocaleString('es-ES')}</div>
              <div className="cl">{hover.point.cluster == null ? 'sin cluster' : `Cluster ${hover.point.cluster}`}</div>
              <div className="zn">‖z‖: {hover.point.znorm.toFixed(2)}</div>
            </div>
          </div>
        )}
      </div>

      <div className="gm-caption">
        Cada punto es una imagen codificada por el encoder. Las distancias reales se calculan sobre z, no
        sobre la proyección 2D.
      </div>
    </Card>
  )
}
