import { Card, CardHeader, Skeleton } from '../ui'
import type { ReconItem } from '../../lib/api'

function Arrow({ color }: { color: string }) {
  return (
    <div className="gm-arrow">
      <svg viewBox="0 0 30 16" fill="none">
        <path d="M3 8h18" stroke={color} strokeWidth="2.2" strokeLinecap="round" />
        <path d="M18 3l6 5-6 5" stroke={color} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

function Pic({ src }: { src?: string }) {
  return (
    <div className="gm-pic">
      {src ? <img src={src} alt="" /> : <Skeleton className="h-full w-full !rounded-none" />}
    </div>
  )
}

interface Props {
  item: ReconItem | null
  latentDim: number
  onInfo?: () => void
}

/** Pipeline del autoencoder con imágenes reales (original / reconstruida / diferencia). */
export function ModelFlowCard({ item, latentDim, onInfo }: Props) {
  return (
    <Card className="flujo">
      <CardHeader title="Flujo del modelo" onInfo={onInfo} infoLabel="Cómo funciona el autoencoder" />
      <div className="gm-pipe">
        <div className="gm-stage">
          <Pic src={item?.original} />
          <div className="cap">
            Original
            <br />
            <span className="mono">64×64×3</span>
          </div>
        </div>
        <Arrow color="#2563EB" />
        <div className="gm-stage">
          <div className="gm-blk enc">
            <div className="ttl">Encoder</div>
            <svg viewBox="0 0 40 30" fill="none">
              <rect x="4" y="16" width="8" height="11" rx="2" fill="#93C0F2" />
              <rect x="16" y="10" width="8" height="17" rx="2" fill="#5B9BE8" />
              <rect x="28" y="4" width="8" height="23" rx="2" fill="#2563EB" />
            </svg>
            <div className="sub">compresión</div>
          </div>
          <div className="cap">
            <span className="mono">64×64×3 → 4×4×256</span>
          </div>
        </div>
        <Arrow color="#2563EB" />
        <div className="gm-stage">
          <div className="gm-neck">
            <span className="gm-zlab">z = {latentDim}</span>
            <i /><i /><i className="sm" /><i /><i className="sm" /><i /><i />
          </div>
          <div className="cap">cuello de botella</div>
        </div>
        <Arrow color="#8B5CF6" />
        <div className="gm-stage">
          <div className="gm-blk dec">
            <div className="ttl">Decoder</div>
            <svg viewBox="0 0 40 30" fill="none">
              <rect x="4" y="16" width="8" height="11" rx="2" fill="#C9B4F0" />
              <rect x="16" y="10" width="8" height="17" rx="2" fill="#A684E0" />
              <rect x="28" y="4" width="8" height="23" rx="2" fill="#8B5CF6" />
            </svg>
            <div className="sub">reconstrucción</div>
          </div>
          <div className="cap">
            <span className="mono">4×4×256 → 64×64×3</span>
          </div>
        </div>
        <Arrow color="#EC4899" />
        <div className="gm-stage">
          <Pic src={item?.reconstruction} />
          <div className="cap">
            Reconstruida
            <br />
            <span className="mono">64×64×3</span>
          </div>
        </div>
        <Arrow color="#EC4899" />
        <div className="gm-stage">
          <Pic src={item?.diff} />
          <div className="cap">
            Diferencia
            <br />
            <span className="mono">|x − x̂|</span>
          </div>
        </div>
      </div>
    </Card>
  )
}
