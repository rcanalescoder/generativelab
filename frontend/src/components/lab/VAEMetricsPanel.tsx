import { useEffect, useState } from 'react'
import { CloseIcon, SparklesIcon } from '../../lib/icons'
import { Spinner } from '../ui'
import { getVAEMetrics, type AEMetrics, type VAEArch } from '../../lib/api-vae'

interface Props {
  /** latent_dim actual (solo para el subtítulo mientras carga). */
  latentDim?: number
  /** variante seleccionada (solo para el subtítulo mientras carga). */
  arch?: VAEArch
  onClose: () => void
}

const ARCH_LABEL: Record<VAEArch, string> = {
  basico: 'Básico',
  grande: 'Grande',
}

const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e))

/** Cifra grande con su unidad y una lectura pedagógica debajo. */
function MetricBig({
  label,
  value,
  unit,
  accent,
  help,
  scale,
}: {
  label: string
  value: string
  unit?: string
  accent: 'green' | 'violet' | 'amber'
  help: string
  scale: string
}) {
  return (
    <div className={`gm-mp-big gm-acc-${accent}`}>
      <div className="gm-mp-big-label">{label}</div>
      <div className="gm-mp-big-num">
        {value}
        {unit && <span className="gm-mp-big-unit">{unit}</span>}
      </div>
      <div className="gm-mp-big-scale">{scale}</div>
      <p className="gm-mp-big-help">{help}</p>
    </div>
  )
}

/** Panel modal de métricas de reconstrucción (PSNR / SSIM / MSE) del VAE.
 *  Reutiliza el markup y las clases de `MetricsPanel` (cableado al AE), apuntando al
 *  endpoint del VAE y a sus etiquetas de variante. La reconstrucción se mide vía μ. */
export function VAEMetricsPanel({ latentDim, arch, onClose }: Props) {
  const [data, setData] = useState<AEMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  useEffect(() => {
    let alive = true
    setLoading(true)
    setError(null)
    getVAEMetrics()
      .then((m) => {
        if (alive) setData(m)
      })
      .catch((e) => {
        if (alive) setError(errMsg(e))
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
  }, [])

  const shownArch = (data?.arch ?? arch) as VAEArch | undefined
  const shownDim = data?.latent_dim ?? latentDim

  return (
    <div
      className="gm-ip-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="gm-ip gm-mp gm-acc-green" role="dialog" aria-modal="true">
        <div className="gm-ip-banner">
          <div className="gm-ip-banner-text">
            <span className="gm-ip-badge">Calidad de reconstrucción</span>
            <h2>Métricas del VAE</h2>
            <p>
              Comparamos cada imagen con su reconstrucción (vía μ) sobre un conjunto de prueba fijo
              {shownArch ? (
                <>
                  {' '}
                  · variante <b>{ARCH_LABEL[shownArch]}</b>
                </>
              ) : null}
              {shownDim != null ? (
                <>
                  {' '}
                  · latent_dim <b>{shownDim}</b>
                </>
              ) : null}
              .
            </p>
          </div>
          <div className="gm-ip-banner-actions">
            <button type="button" className="gm-ip-x" onClick={onClose} aria-label="Cerrar">
              <CloseIcon />
            </button>
          </div>
        </div>

        <div className="gm-ip-scroll">
          {loading && (
            <div className="gm-mp-state">
              <Spinner />
              <span>Midiendo la reconstrucción sobre el conjunto de prueba…</span>
            </div>
          )}

          {error && !loading && (
            <div className="gm-ip-note tone-warn">
              No se pudieron calcular las métricas: {error}
              <br />
              ¿Has entrenado o cargado el checkpoint demo de esta variante?
            </div>
          )}

          {data && !loading && !error && (
            <>
              {/* cifras grandes */}
              <div className="gm-mp-grid">
                <MetricBig
                  label="PSNR"
                  value={data.psnr.toFixed(2)}
                  unit=" dB"
                  accent="green"
                  scale="0 → ∞ (más alto, mejor)"
                  help="Relación señal-ruido de pico: cuánto se aleja la reconstrucción del original en escala logarítmica. Por encima de 30 dB es excelente; ~25 dB es decente; alrededor de 18 dB la imagen se ve pobre y borrosa."
                />
                <MetricBig
                  label="SSIM"
                  value={data.ssim.toFixed(3)}
                  accent="violet"
                  scale="0 → 1 (más alto, mejor)"
                  help="Índice de similitud estructural: compara luminancia, contraste y estructura como lo haría el ojo. Cerca de 1 significa que se preservó la estructura; por debajo de ~0,7 se nota pérdida de detalle."
                />
                <MetricBig
                  label="MSE"
                  value={data.mse.toFixed(4)}
                  accent="amber"
                  scale="0 → 1 (más bajo, mejor)"
                  help="Error cuadrático medio por píxel en [0,1]: el promedio de las diferencias al cuadrado. En el VAE es el término de reconstrucción de la pérdida; cuanto más cerca de 0, más fiel la reconstrucción."
                />
              </div>

              <div className="gm-ip-note tone-info">
                Promedios sobre <b>{data.n}</b> imágenes de prueba que el modelo no usa para ajustar
                estas cifras. Se mide la reconstrucción determinista (vía μ, sin muestrear el ruido),
                para reflejar la calidad real del modelo. PSNR penaliza el error píxel a píxel; SSIM
                se fija en la estructura percibida.
              </div>

              {/* ejemplos */}
              {data.examples.length > 0 && (
                <section className="gm-ip-sec gm-acc-green">
                  <div className="gm-ip-sec-head">
                    <span className="ic">
                      <SparklesIcon />
                    </span>
                    <h3>Ejemplos · original vs reconstruida vs diferencia</h3>
                  </div>
                  <div className="gm-mp-examples">
                    {data.examples.map((ex) => (
                      <figure key={ex.id} className="gm-mp-ex">
                        <div className="gm-mp-ex-row">
                          <div className="gm-mp-ex-cell">
                            <img src={ex.original} alt={`original ${ex.id}`} />
                            <figcaption>original</figcaption>
                          </div>
                          <div className="gm-mp-ex-cell">
                            <img src={ex.reconstruction} alt={`reconstrucción ${ex.id}`} />
                            <figcaption>reconstruida</figcaption>
                          </div>
                          <div className="gm-mp-ex-cell">
                            <img src={ex.diff} alt={`diferencia ${ex.id}`} />
                            <figcaption>|x − x̂|</figcaption>
                          </div>
                        </div>
                        <div className="gm-mp-ex-stats">
                          <span>
                            PSNR <b className="mono">{ex.psnr.toFixed(1)}</b> dB
                          </span>
                          <span>
                            SSIM <b className="mono">{ex.ssim.toFixed(3)}</b>
                          </span>
                        </div>
                      </figure>
                    ))}
                  </div>
                </section>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
