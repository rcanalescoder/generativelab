import { useEffect, type ReactNode } from 'react'
import { CloseIcon, LayersIcon } from '../../lib/icons'
import { cn } from '../../lib/cn'
import type { ArchColumn, ArchLatent, ArchLayer, Architecture } from '../../content/architectures'

interface Props {
  /** Esquema de la red a dibujar (de content/architectures.ts). */
  arch: Architecture
  /** latent_dim actual del modelo; sustituye el marcador {DIM} donde aplique. */
  latentDim?: number
  onClose: () => void
}

/** Sustituye el marcador {DIM} por el latent_dim actual (o "latent_dim" si no se pasa). */
function withDim(text: string, latentDim?: number): string {
  return text.replace(/\{DIM\}/g, latentDim != null ? String(latentDim) : 'latent_dim')
}

function LayerBox({ layer, latentDim }: { layer: ArchLayer; latentDim?: number }) {
  return (
    <div className={cn('gm-arch-box', layer.destacada && 'is-edge')}>
      <div className="gm-arch-box-name">{layer.nombre}</div>
      {layer.detalle && <div className="gm-arch-box-detail">{withDim(layer.detalle, latentDim)}</div>}
      <div className="gm-arch-shape">{withDim(layer.forma, latentDim)}</div>
    </div>
  )
}

function Column({ col, latentDim }: { col: ArchColumn; latentDim?: number }) {
  return (
    <div className={cn('gm-arch-col', `gm-acc-${col.acento}`)}>
      <div className="gm-arch-col-head">
        <span className="gm-arch-col-title">{col.titulo}</span>
        {col.subtitulo && <span className="gm-arch-col-sub">{col.subtitulo}</span>}
      </div>
      <div className="gm-arch-stack">
        {col.capas.map((layer, i) => (
          <div key={i} className="gm-arch-item">
            <LayerBox layer={layer} latentDim={latentDim} />
            {i < col.capas.length - 1 && (
              <span className="gm-arch-arrow" aria-hidden="true">
                ↓
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function LatentBlock({ latent, latentDim }: { latent: ArchLatent; latentDim?: number }) {
  return (
    <div className={cn('gm-arch-latent', `gm-acc-${latent.acento}`)}>
      <div className="gm-arch-latent-title">{withDim(latent.titulo, latentDim)}</div>
      <div className="gm-arch-latent-lines">
        {latent.lineas.map((line, i) => (
          <span key={i} className="gm-arch-latent-line">
            {withDim(line, latent.usaLatentDim ? latentDim : undefined)}
          </span>
        ))}
      </div>
    </div>
  )
}

export function ArchitectureModal({ arch, latentDim, onClose }: Props) {
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

  // Entre columnas: para AE/VAE intercalamos el bloque latente (encoder → z → decoder);
  // para el resto, una flecha simple que conecta las columnas.
  const cols = arch.columnas

  return (
    <div
      className="gm-ip-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className="gm-ip gm-arch gm-acc-violet" role="dialog" aria-modal="true">
        <div className="gm-ip-banner">
          <div className="gm-ip-banner-text">
            <span className="gm-ip-badge">Estructura de la red</span>
            <h2>{arch.titulo}</h2>
            <p>{arch.resumen}</p>
          </div>
          <div className="gm-ip-banner-actions">
            <button type="button" className="gm-ip-x" onClick={onClose} aria-label="Cerrar">
              <CloseIcon />
            </button>
          </div>
        </div>

        <div className="gm-ip-scroll">
          <div className="gm-arch-meta">
            <span className="gm-arch-meta-ic">
              <LayersIcon />
            </span>
            <span>{arch.entrada}</span>
          </div>

          <div className="gm-arch-diagram">
            {cols.map((col, i) => {
              const isLast = i === cols.length - 1
              // Conector entre columna i e i+1.
              let connector: ReactNode = null
              if (!isLast) {
                if (arch.latente && i === 0) {
                  connector = <LatentBlock latent={arch.latente} latentDim={latentDim} />
                } else {
                  connector = (
                    <div className="gm-arch-connector" aria-hidden="true">
                      <span className="gm-arch-connector-arrow">→</span>
                    </div>
                  )
                }
              }
              return (
                <div key={i} className="gm-arch-group">
                  <Column col={col} latentDim={latentDim} />
                  {connector}
                </div>
              )
            })}
          </div>

          {arch.notas && arch.notas.length > 0 && (
            <ul className="gm-arch-notes">
              {arch.notas.map((nota, i) => (
                <li key={i}>{nota}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  )
}
