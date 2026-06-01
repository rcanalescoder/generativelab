import { useEffect, useRef, useState, type ReactNode } from 'react'
import { Button, SegmentedControl, Slider } from '../ui'
import { CloseIcon, GridIcon, PlayIcon } from '../../lib/icons'
import { cn } from '../../lib/cn'
import {
  cancelGridSearch,
  gridSearch,
  type GridConfig,
  type GridEvent,
  type GridRequest,
  type GridResult,
} from '../../lib/api'

const LD_OPTS = [32, 64, 128, 192]
const LR_OPTS = [0.0003, 0.001, 0.003]
const LOSS_OPTS: Array<'mse' | 'l1'> = ['mse', 'l1']

interface Props {
  seed: number
  onClose: () => void
  onApply: (cfg: GridConfig) => void
  /** Función de búsqueda (por defecto la del Autoencoder). Permite reutilizar el modal en VAE. */
  runSearch?: (body: GridRequest, onEvent: (ev: GridEvent) => void, signal?: AbortSignal) => Promise<void>
  cancelSearch?: () => Promise<unknown>
  title?: string
}

function Chip({ active, onClick, children }: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button type="button" className={cn('gm-chip', active && 'is-sel')} onClick={onClick}>
      {children}
    </button>
  )
}

export function GridSearchModal({
  seed,
  onClose,
  onApply,
  runSearch = gridSearch,
  cancelSearch = cancelGridSearch,
  title,
}: Props) {
  const [scope, setScope] = useState<'quick' | 'full'>('quick')
  const [epochs, setEpochs] = useState(3)
  const [lds, setLds] = useState<number[]>([32, 64, 128])
  const [lrs, setLrs] = useState<number[]>([0.001])
  const [losses, setLosses] = useState<Array<'mse' | 'l1'>>(['mse'])

  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<GridResult[]>([])
  const [best, setBest] = useState<GridResult | null>(null)
  const [progress, setProgress] = useState<{ index: number; total: number; epoch: number; epochs: number } | null>(null)
  const [total, setTotal] = useState(0)
  const [done, setDone] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const combos = Math.min(16, lds.length * lrs.length * losses.length)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !running) onClose()
    }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose, running])

  const toggle = <T,>(list: T[], v: T, set: (l: T[]) => void) =>
    set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v])

  const run = () => {
    if (!combos) return
    setRunning(true)
    setResults([])
    setBest(null)
    setDone(false)
    setError(null)
    setProgress(null)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    runSearch(
      { scope, epochs, seed, grid: { latent_dim: lds, learning_rate: lrs, loss: losses } },
      (ev) => {
        if (ev.type === 'start') setTotal(ev.total)
        else if (ev.type === 'progress') setProgress({ index: ev.index, total: ev.total, epoch: ev.epoch, epochs: ev.epochs })
        else if (ev.type === 'trial') {
          setResults((prev) => [...prev, ev.result].sort((a, b) => a.eval_mse - b.eval_mse))
          setBest(ev.best)
        } else if (ev.type === 'done') {
          setResults(ev.results)
          setBest(ev.best)
          setDone(true)
        } else if (ev.type === 'error') setError(ev.message)
      },
      ctrl.signal,
    )
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(e instanceof Error ? e.message : String(e))
      })
      .finally(() => {
        setRunning(false)
        abortRef.current = null
      })
  }

  const cancel = () => {
    cancelSearch().catch(() => undefined)
    abortRef.current?.abort()
  }

  const pct = progress ? Math.round(((progress.index + progress.epoch / progress.epochs) / progress.total) * 100) : 0

  return (
    <div className="gm-ip-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget && !running) onClose() }}>
      <div className="gm-ip gm-acc-cyan" role="dialog" aria-modal="true">
        <div className="gm-ip-banner">
          <div className="gm-ip-banner-text">
            <span className="gm-ip-badge">Búsqueda de hiperparámetros</span>
            <h2>{title ?? 'Buscar mejores parámetros'}</h2>
            <p>Prueba varias combinaciones y mide la calidad de reconstrucción (MSE) sobre un conjunto de prueba fijo. Gana la de menor error.</p>
          </div>
          <div className="gm-ip-banner-actions">
            <button type="button" className="gm-ip-x" onClick={() => (running ? cancel() : onClose())} aria-label="Cerrar">
              <CloseIcon />
            </button>
          </div>
        </div>

        <div className="gm-ip-scroll">
          {/* configuración */}
          <section className="gm-ip-sec gm-acc-cyan">
            <div className="gm-ip-sec-head">
              <span className="ic"><GridIcon /></span>
              <h3>Rejilla a probar</h3>
            </div>

            <div className="mb-3">
              <div className="mb-1.5 text-[12.5px] font-medium text-[color:var(--slate)]">datos de entrenamiento</div>
              <SegmentedControl
                value={scope}
                onChange={setScope}
                options={[
                  { label: 'Rápido (~6k)', value: 'quick' },
                  { label: 'Completo (todo)', value: 'full' },
                ]}
              />
            </div>

            <Slider label="epochs por combinación" value={epochs} min={1} max={8} onChange={setEpochs} />

            <div className="mt-2 flex flex-col gap-2.5">
              <div className="flex items-center gap-2.5">
                <span className="w-[96px] font-mono text-[12.5px] text-[color:var(--slate)]">latent_dim</span>
                <div className="flex flex-wrap gap-1.5">
                  {LD_OPTS.map((v) => (
                    <Chip key={v} active={lds.includes(v)} onClick={() => toggle(lds, v, setLds)}>{v}</Chip>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="w-[96px] font-mono text-[12.5px] text-[color:var(--slate)]">learning_rate</span>
                <div className="flex flex-wrap gap-1.5">
                  {LR_OPTS.map((v) => (
                    <Chip key={v} active={lrs.includes(v)} onClick={() => toggle(lrs, v, setLrs)}>{v}</Chip>
                  ))}
                </div>
              </div>
              <div className="flex items-center gap-2.5">
                <span className="w-[96px] font-mono text-[12.5px] text-[color:var(--slate)]">loss</span>
                <div className="flex flex-wrap gap-1.5">
                  {LOSS_OPTS.map((v) => (
                    <Chip key={v} active={losses.includes(v)} onClick={() => toggle(losses, v, setLosses)}>{v.toUpperCase()}</Chip>
                  ))}
                </div>
              </div>
            </div>

            <div className="gm-ip-note tone-info mt-3">
              {combos} combinación{combos === 1 ? '' : 'es'} × {epochs} epoch{epochs === 1 ? '' : 's'}
              {scope === 'full' ? ' sobre TODO el dataset (puede tardar varios minutos).' : ' sobre un subconjunto rápido.'}
            </div>
          </section>

          {/* ejecución */}
          {!running && !done && (
            <Button variant="primary" full icon={<PlayIcon />} onClick={run} disabled={combos === 0}>
              Empezar búsqueda
            </Button>
          )}
          {running && (
            <div>
              <div className="mb-2 flex items-center justify-between text-[13px] text-[color:var(--slate)]">
                <span>
                  Combinación {progress ? progress.index + 1 : 1}/{total || combos}
                  {progress ? ` · epoch ${progress.epoch}/${progress.epochs}` : ''}
                </span>
                <button className="gm-btn is-ghost" onClick={cancel}>Cancelar</button>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-[color:var(--line)]">
                <div className="h-full rounded-full bg-[color:var(--acc)] transition-[width]" style={{ width: `${pct}%` }} />
              </div>
            </div>
          )}
          {error && <div className="gm-ip-note tone-warn">{error}</div>}

          {/* resultados */}
          {results.length > 0 && (
            <section className="gm-ip-sec gm-acc-cyan">
              <div className="gm-ip-sec-head">
                <h3>Resultados {done ? '(ordenados por error)' : '(en curso…)'}</h3>
              </div>
              <table className="gm-gs-table">
                <thead>
                  <tr>
                    <th>#</th><th>latent_dim</th><th>lr</th><th>loss</th><th>epochs</th><th>MSE eval</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => {
                    const isBest = best && r.latent_dim === best.latent_dim && r.learning_rate === best.learning_rate && r.loss === best.loss
                    return (
                      <tr key={i} className={isBest ? 'best' : undefined}>
                        <td>{i + 1}</td>
                        <td className="mono">{r.latent_dim}</td>
                        <td className="mono">{r.learning_rate}</td>
                        <td className="mono">{r.loss.toUpperCase()}</td>
                        <td className="mono">{r.epochs}</td>
                        <td className="mono">{r.eval_mse.toFixed(4)}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </section>
          )}

          {/* aplicar */}
          {done && best && (
            <div className="flex items-center gap-2.5">
              <div className="flex-1 text-[13px] text-[color:var(--slate)]">
                Mejor: <span className="mono text-[color:var(--ink)]">latent_dim={best.latent_dim}, lr={best.learning_rate}, {best.loss.toUpperCase()}</span>
                {' '}· MSE <span className="mono">{best.eval_mse.toFixed(4)}</span>
              </div>
              <Button variant="ghost" onClick={onClose}>Cerrar</Button>
              <Button variant="primary" onClick={() => { onApply(best); onClose() }}>Aplicar mejores</Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
