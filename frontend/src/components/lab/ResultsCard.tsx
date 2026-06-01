import type { CSSProperties } from 'react'
import { Card, CardHeader, Skeleton, Slider } from '../ui'
import type { InterpResult, NeighborsResult } from '../../lib/api'

interface Props {
  neighbors: NeighborsResult | null
  interp: InterpResult | null
  alpha: number
  onAlpha: (v: number) => void
  onInfo?: () => void
}

export function ResultsCard({ neighbors, interp, alpha, onAlpha, onInfo }: Props) {
  const steps = interp?.frames.length ?? 6
  const gridStyle = { gridTemplateColumns: `repeat(${steps}, 1fr)` } as CSSProperties
  const highlighted = interp ? Math.round(alpha * (steps - 1)) : -1

  return (
    <Card rise={5} className="results">
      <CardHeader title="Resultados del espacio latente" onInfo={onInfo} infoLabel="Vecinos e interpolación" />
      <div className="gm-results-grid">
        {/* Vecinos */}
        <div>
          <div className="gm-sub-h">Vecinos similares</div>
          <div className="gm-sub-d">Búsqueda por distancia en el espacio latente</div>
          <div className="gm-neighbors">
            {!neighbors
              ? Array.from({ length: 6 }).map((_, i) => (
                  <div className="gm-nb" key={i}>
                    <Skeleton className="aspect-square w-full !rounded-[12px]" />
                    <span className="cap mono">·</span>
                  </div>
                ))
              : [
                  <div className="gm-nb query" key="q">
                    <div className="pic">
                      <img src={neighbors.query.image} alt="consulta" />
                    </div>
                    <span className="cap">Consulta</span>
                  </div>,
                  ...neighbors.neighbors.map((nb) => (
                    <div className="gm-nb" key={nb.id}>
                      <div className="pic">
                        <img src={nb.image} alt={`vecino ${nb.id}`} />
                      </div>
                      <span className="cap mono">{nb.distance.toFixed(2)}</span>
                    </div>
                  )),
                ]}
          </div>
        </div>

        <div className="vsep" />

        {/* Interpolación */}
        <div>
          <div className="gm-interrow">
            <span className="gm-sub-h" style={{ margin: 0 }}>
              Interpolación A → B
            </span>
            <span className="gm-formula">
              z<sub>α</sub> = (1−α)·z<sub>A</sub> + α·z<sub>B</sub>
            </span>
          </div>
          <div className="gm-interp" style={gridStyle}>
            {!interp
              ? Array.from({ length: steps }).map((_, i) => (
                  <div className="gm-ip" key={i}>
                    <Skeleton className="aspect-square w-full !rounded-[12px]" />
                  </div>
                ))
              : interp.frames.map((f, i) => (
                  <div className="gm-ip" key={i}>
                    <div
                      className="pic"
                      style={i === highlighted ? { boxShadow: '0 0 0 2.5px var(--blue)' } : undefined}
                    >
                      <img src={f.image} alt={`α=${f.alpha}`} />
                    </div>
                    <span className="cap">{i === 0 ? 'A' : i === steps - 1 ? 'B' : ''}</span>
                  </div>
                ))}
          </div>
          <div className="gm-alpha">
            <span className="lbl">α</span>
            <div className="flex-1">
              <Slider value={Math.round(alpha * 100)} min={0} max={100} onChange={(v) => onAlpha(v / 100)} />
              <div className="scale">
                <span>0</span>
                <span>0.50</span>
                <span>1</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}
