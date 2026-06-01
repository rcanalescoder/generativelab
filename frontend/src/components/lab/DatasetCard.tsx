import { useRef } from 'react'
import { Card, CardHeader, Skeleton } from '../ui'
import { DatabaseIcon, ImageIcon, RefreshIcon, UploadIcon } from '../../lib/icons'
import type { DatasetInfo, Sample } from '../../lib/api'

interface Props {
  info: DatasetInfo | null
  samples: Sample[]
  onShuffle: () => void
  onUpload: (file: File) => void
  onSelect?: (id: number) => void
  onInfo?: () => void
  busy?: boolean
}

export function DatasetCard({ info, samples, onShuffle, onUpload, onSelect, onInfo, busy }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)

  return (
    <Card rise={1}>
      <CardHeader title="Dataset visual" onInfo={onInfo} infoLabel="Sobre el dataset" />
      <div className="gm-ds-name">
        <span className="db">
          <DatabaseIcon />
        </span>
        Anime Faces
      </div>

      <div className="gm-gallery">
        {samples.length === 0
          ? Array.from({ length: 8 }).map((_, i) => <Skeleton key={i} className="aspect-square rounded-[12px]" />)
          : samples.slice(0, 8).map((s) => (
              <div
                key={s.id}
                className="gm-thumb cursor-pointer transition-transform hover:-translate-y-0.5"
                onClick={() => onSelect?.(s.id)}
                title={`Usar la muestra ${s.id} en el flujo`}
              >
                <img src={s.image} alt={`muestra ${s.id}`} loading="lazy" />
              </div>
            ))}
      </div>

      <div className="gm-ds-meta">
        {info ? (
          <>
            <span className="mono">{info.count.toLocaleString('es-ES')}</span> imágenes ·{' '}
            <span className="mono">64×64 RGB</span> · sin etiquetas
          </>
        ) : (
          'cargando metadatos…'
        )}
      </div>

      <div className="gm-ds-btns">
        <button className="gm-chip is-sel" onClick={onShuffle} disabled={busy}>
          <ImageIcon />
          Ver muestras
        </button>
        <button className="gm-chip" onClick={() => fileRef.current?.click()} disabled={busy}>
          <UploadIcon />
          Subir imagen
        </button>
        <button className="gm-chip" disabled title="Próximamente">
          <RefreshIcon />
          Cambiar dataset
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onUpload(f)
            e.target.value = ''
          }}
        />
      </div>
    </Card>
  )
}
