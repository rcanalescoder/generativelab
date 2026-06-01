import type { CSSProperties } from 'react'
import { Button, Card, CardHeader, Skeleton, Slider } from '../ui'
import { RefreshIcon } from '../../lib/icons'
import type { InterpResult, NeighborsResult } from '../../lib/api'

interface Props {
  neighbors: NeighborsResult | null
  interp: InterpResult | null
  alpha: number
  onAlpha: (v: number) => void
  /** Imágenes originales de los extremos A y B (para que quede claro qué son). */
  aImage?: string
  bImage?: string
  onChangeB?: () => void
  busy?: boolean
  onInfo?: () => void
}

export function ResultsCard({
  neighbors,
  interp,
  alpha,
  onAlpha,
  aImage,
  bImage,
  onChangeB,
  busy,
  onInfo,
}: Props) {
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
          <div className="gm-sub-d">
            Las caras del dataset cuyo código <span className="mono">z</span> es más parecido al de la
            consulta (la primera, en rosa). El número es la distancia: cuanto menor, más parecida según el modelo.
          </div>
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
          <div className="gm-sub-d">
            Tomamos dos caras —<b>A</b> y <b>B</b>— y generamos pasos intermedios mezclando sus códigos.
            Cada fotograma es una cara nueva decodificada.
          </div>

          {/* extremos A y B (qué son exactamente) */}
          <div className="gm-ab">
            <div className="ep">
              <div className="pic">
                {aImage ? <img src={aImage} alt="A" /> : <Skeleton className="h-full w-full !rounded-[10px]" />}
              </div>
              <span className="lab">
                <b>A</b> · origen
              </span>
            </div>
            <span className="arrow">→</span>
            <div className="ep">
              <div className="pic">
                {bImage ? <img src={bImage} alt="B" /> : <Skeleton className="h-full w-full !rounded-[10px]" />}
              </div>
              <span className="lab">
                <b>B</b> · destino
              </span>
            </div>
            {onChangeB && (
              <Button variant="ghost" icon={<RefreshIcon />} onClick={onChangeB} disabled={busy}>
                Cambiar B
              </Button>
            )}
          </div>

          {/* tira de fotogramas (mezcla decodificada) */}
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
                    <span className="cap">{i === 0 ? 'A' : i === steps - 1 ? 'B' : `α ${f.alpha}`}</span>
                  </div>
                ))}
          </div>

          {/* control α */}
          <div className="gm-alpha">
            <span className="lbl">α = {alpha.toFixed(2)}</span>
            <div className="flex-1">
              <Slider value={Math.round(alpha * 100)} min={0} max={100} onChange={(v) => onAlpha(v / 100)} />
              <div className="scale">
                <span>A (0)</span>
                <span>mezcla</span>
                <span>B (1)</span>
              </div>
            </div>
          </div>
          <div className="gm-sub-d" style={{ marginTop: 9 }}>
            Mueve <span className="mono">α</span> para recorrer la transición: en 0 ves A, en 1 ves B, y en
            medio la mezcla (se resalta el fotograma correspondiente).
          </div>
        </div>
      </div>
    </Card>
  )
}
