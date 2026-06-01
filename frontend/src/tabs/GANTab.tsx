import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Card, CardHeader, SegmentedControl, Skeleton, Slider, StatusLine, Tag } from '../components/ui'
import { TrainingChart } from '../components/lab/TrainingChart'
import { useInfo } from '../components/info/InfoProvider'
import { InterpolateIcon, PlayIcon, RefreshIcon, SparklesIcon } from '../lib/icons'
import type { ModelContent } from '../content/types'
import ganContent from '../content/gan.json'
import {
  cancelGANTraining,
  generateGAN,
  getGANStatus,
  interpolateGAN,
  trainGAN,
  type GANInterpResult,
  type GANLossPoint,
  type GANStatus,
} from '../lib/api-gan'

const C = ganContent as unknown as ModelContent

const LR_VALUES = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]
const nearestLrIdx = (lr: number) => {
  let best = 0
  for (let i = 1; i < LR_VALUES.length; i++) {
    if (Math.abs(LR_VALUES[i] - lr) < Math.abs(LR_VALUES[best] - lr)) best = i
  }
  return best
}
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e))

/** Valores por defecto (los que ves al abrir la app, ≈ checkpoint demo). */
const DEFAULTS = {
  zDim: 100,
  lrIdx: 2, // 0.0002
  epochs: 25,
  nSamples: 16,
  genSeed: 42,
  interpSteps: 8,
  interpSeed: 7,
  trainSeed: 42,
  mode: 'quick' as const,
}

export function GANTab() {
  const { open } = useInfo()
  const [status, setStatus] = useState<GANStatus | null>(null)

  // galería generada
  const [gallery, setGallery] = useState<string[] | null>(null)
  const [nSamples, setNSamples] = useState(DEFAULTS.nSamples)
  const [genSeed, setGenSeed] = useState(DEFAULTS.genSeed)

  // interpolación en z
  const [interp, setInterp] = useState<GANInterpResult | null>(null)
  const [interpSteps, setInterpSteps] = useState(DEFAULTS.interpSteps)
  const [interpSeed, setInterpSeed] = useState(DEFAULTS.interpSeed)

  // hiperparámetros del modelo (requieren reentrenar)
  const [zDim, setZDim] = useState(DEFAULTS.zDim)
  const [lrIdx, setLrIdx] = useState(DEFAULTS.lrIdx)
  const [epochs, setEpochs] = useState(DEFAULTS.epochs)
  const [trainSeed, setTrainSeed] = useState(DEFAULTS.trainSeed)
  const [mode, setMode] = useState<'quick' | 'full'>(DEFAULTS.mode)

  // entrenamiento en vivo
  const [training, setTraining] = useState(false)
  const [liveLoss, setLiveLoss] = useState<GANLossPoint[]>([])
  const [trainInfo, setTrainInfo] = useState<{ epoch: number; epochs: number; g: number; d: number } | null>(null)
  const [preview, setPreview] = useState<string[] | null>(null)

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const lr = LR_VALUES[lrIdx]
  const trained = !!status?.trained
  const outdated =
    trained &&
    !!status &&
    (zDim !== status.hyperparams.z_dim ||
      lr !== status.hyperparams.learning_rate ||
      epochs !== status.hyperparams.epochs)

  const run = useCallback(async (fn: () => Promise<void>) => {
    setBusy(true)
    setError(null)
    try {
      await fn()
    } catch (e) {
      setError(errMsg(e))
    } finally {
      setBusy(false)
    }
  }, [])

  // carga inicial: status y, si está entrenado, una generación y una interpolación
  useEffect(() => {
    ;(async () => {
      try {
        const st = await getGANStatus()
        setStatus(st)
        setZDim(st.hyperparams.z_dim)
        setEpochs(st.hyperparams.epochs)
        setLrIdx(nearestLrIdx(st.hyperparams.learning_rate))
        setTrainSeed(st.seed)
        if (st.trained) {
          setGallery(await generateGAN(DEFAULTS.nSamples, DEFAULTS.genSeed))
          setInterp(await interpolateGAN(DEFAULTS.interpSteps, DEFAULTS.interpSeed))
        }
      } catch (e) {
        setError(errMsg(e))
      }
    })()
  }, [])

  const doGenerate = () =>
    run(async () => {
      setGallery(await generateGAN(nSamples, genSeed))
    })

  const shuffleGenerate = () =>
    run(async () => {
      const s = genSeed + 1
      setGenSeed(s)
      setGallery(await generateGAN(nSamples, s))
    })

  const doInterpolate = () =>
    run(async () => {
      setInterp(await interpolateGAN(interpSteps, interpSeed))
    })

  const shuffleInterpolate = () =>
    run(async () => {
      const s = interpSeed + 1
      setInterpSeed(s)
      setInterp(await interpolateGAN(interpSteps, s))
    })

  const onTrain = () => {
    setError(null)
    setTraining(true)
    setLiveLoss([])
    setTrainInfo(null)
    setPreview(null)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    trainGAN(
      { mode, seed: trainSeed, hyperparams: { z_dim: zDim, learning_rate: lr, epochs } },
      (ev) => {
        if (ev.type === 'epoch') {
          setLiveLoss((prev) => [...prev, { epoch: ev.epoch, g_loss: ev.g_loss, d_loss: ev.d_loss }])
          setTrainInfo({ epoch: ev.epoch, epochs: ev.epochs, g: ev.g_loss, d: ev.d_loss })
          setPreview(ev.preview)
        } else if (ev.type === 'done') {
          setLiveLoss(ev.loss_history)
        } else if (ev.type === 'error') {
          setError(ev.message)
        }
      },
      ctrl.signal,
    )
      .catch((e) => {
        if (!ctrl.signal.aborted) setError(errMsg(e))
      })
      .finally(() => {
        setTraining(false)
        abortRef.current = null
        // refrescar estado y paneles con el generador recién entrenado
        run(async () => {
          const st = await getGANStatus()
          setStatus(st)
          if (st.trained) {
            setGallery(await generateGAN(nSamples, genSeed))
            setInterp(await interpolateGAN(interpSteps, interpSeed))
          }
        })
      })
  }

  const onCancel = () => {
    cancelGANTraining().catch(() => undefined)
    abortRef.current?.abort()
  }

  const resetDefaults = () => {
    setZDim(DEFAULTS.zDim)
    setLrIdx(DEFAULTS.lrIdx)
    setEpochs(DEFAULTS.epochs)
    setTrainSeed(DEFAULTS.trainSeed)
    setMode(DEFAULTS.mode)
  }

  // curvas: el GAN tiene dos pérdidas; las mapeamos a {epoch, loss} para TrainingChart.
  const gPoints = liveLoss.map((p) => ({ epoch: p.epoch, loss: p.g_loss }))
  const dPoints = liveLoss.map((p) => ({ epoch: p.epoch, loss: p.d_loss }))

  const statusLine = training ? (
    trainInfo ? (
      <>
        epoch <b>{trainInfo.epoch}/{trainInfo.epochs}</b> · g_loss{' '}
        <span className="mono">{trainInfo.g.toFixed(4)}</span> · d_loss{' '}
        <span className="mono">{trainInfo.d.toFixed(4)}</span>
      </>
    ) : (
      <>iniciando entrenamiento…</>
    )
  ) : outdated ? (
    <span style={{ color: '#B5740B' }}>cambios sin aplicar — reentrena para verlos</span>
  ) : trained && status?.loss_history.length ? (
    <>
      modelo entrenado · g_loss{' '}
      <span className="mono">{status.loss_history[status.loss_history.length - 1].g_loss.toFixed(4)}</span> · d_loss{' '}
      <span className="mono">{status.loss_history[status.loss_history.length - 1].d_loss.toFixed(4)}</span>
    </>
  ) : (
    <>sin entrenar — entrena o carga el checkpoint demo</>
  )

  const galleryCols =
    nSamples <= 4 ? 'grid-cols-2' : nSamples <= 9 ? 'grid-cols-3' : nSamples <= 16 ? 'grid-cols-4' : 'grid-cols-5'

  const emptyHint = (
    <div className="flex aspect-square w-full items-center justify-center rounded-[14px] border border-dashed border-[color:var(--line)] text-center text-[12.5px] text-[color:var(--muted)]">
      entrena o carga el checkpoint demo
    </div>
  )

  return (
    <>
      {error && (
        <div
          className="mb-[18px] flex items-center justify-between rounded-[12px] border px-4 py-3 text-[13.5px]"
          style={{ background: 'var(--pink-soft)', borderColor: '#F6C9DD', color: '#9D2A5E' }}
        >
          <span>{error}</span>
          <button className="gm-btn is-ghost" onClick={() => setError(null)}>
            cerrar
          </button>
        </div>
      )}

      <div className="grid items-start gap-[22px] grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(336px,388px)]">
        {/* IZQUIERDA / CENTRO: galería + interpolación */}
        <div className="flex min-w-0 flex-col gap-[22px]">
          {/* Galería generada */}
          <Card rise={1}>
            <CardHeader
              title="Galería generada"
              actions={<Tag variant="live">interactivo</Tag>}
              onInfo={() => open(C.topics.generacion)}
              infoLabel="Sobre la generación"
            />
            <p className="mb-[14px] text-[13px] leading-snug text-[color:var(--slate)]">
              Cada cara nace de un vector de ruido <span className="mono">z ~ N(0,1)</span> que el generador convierte
              en imagen. Ninguna existe en el dataset.
            </p>

            <div className={`mb-[16px] grid gap-[9px] ${galleryCols}`}>
              {gallery === null
                ? Array.from({ length: nSamples }).map((_, i) =>
                    trained ? (
                      <Skeleton key={i} className="aspect-square w-full !rounded-[14px]" />
                    ) : (
                      <div key={i}>{emptyHint}</div>
                    ),
                  )
                : gallery.map((img, i) => (
                    <div key={i} className="gm-thumb !rounded-[14px]">
                      <img src={img} alt={`muestra generada ${i}`} loading="lazy" />
                    </div>
                  ))}
            </div>

            <div className="grid grid-cols-1 gap-[11px] sm:grid-cols-2">
              <Slider label="nº de muestras" value={nSamples} min={4} max={25} onChange={setNSamples} />
              <div className="gm-ctrl">
                <div className="row">
                  <span className="name">seed</span>
                  <input
                    type="number"
                    className="gm-val"
                    style={{ width: 72 }}
                    value={genSeed}
                    onChange={(e) => setGenSeed(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <div className="mt-[6px] grid grid-cols-2 gap-[11px]">
              <Button icon={<SparklesIcon />} onClick={doGenerate} disabled={!trained || busy || training}>
                Generar
              </Button>
              <Button icon={<RefreshIcon />} onClick={shuffleGenerate} disabled={!trained || busy || training}>
                Generar nuevas
              </Button>
            </div>
          </Card>

          {/* Interpolación en z */}
          <Card rise={2}>
            <CardHeader
              title="Interpolación en z"
              actions={<Tag variant="live">interactivo</Tag>}
              onInfo={() => open(C.topics.interpolacion)}
              infoLabel="Sobre la interpolación"
            />
            <div className="mb-[12px] flex flex-wrap items-center justify-between gap-2">
              <p className="max-w-[60ch] text-[13px] leading-snug text-[color:var(--slate)]">
                Caminamos por el espacio de ruido entre dos puntos aleatorios. Como la GAN no tiene encoder, los
                extremos son ruido, no caras del dataset.
              </p>
              <span className="gm-formula whitespace-nowrap">
                z<sub>α</sub> = (1−α)·z<sub>A</sub> + α·z<sub>B</sub>
              </span>
            </div>

            <div
              className="mb-[14px] grid gap-[8px]"
              style={{ gridTemplateColumns: `repeat(${interp?.frames.length ?? interpSteps}, minmax(0, 1fr))` }}
            >
              {interp === null
                ? Array.from({ length: interpSteps }).map((_, i) =>
                    trained ? (
                      <Skeleton key={i} className="aspect-square w-full !rounded-[12px]" />
                    ) : (
                      <div key={i}>{emptyHint}</div>
                    ),
                  )
                : interp.frames.map((f, i) => (
                    <div key={i} className="flex flex-col items-center gap-1">
                      <div className="gm-thumb w-full !rounded-[12px]">
                        <img src={f.image} alt={`α=${f.alpha}`} loading="lazy" />
                      </div>
                      <span className="font-mono text-[10.5px] text-[color:var(--muted)]">
                        {i === 0 ? 'A' : i === interp.frames.length - 1 ? 'B' : f.alpha}
                      </span>
                    </div>
                  ))}
            </div>

            <div className="grid grid-cols-1 gap-[11px] sm:grid-cols-2">
              <Slider label="pasos" value={interpSteps} min={3} max={12} onChange={setInterpSteps} />
              <div className="gm-ctrl">
                <div className="row">
                  <span className="name">seed</span>
                  <input
                    type="number"
                    className="gm-val"
                    style={{ width: 72 }}
                    value={interpSeed}
                    onChange={(e) => setInterpSeed(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <div className="mt-[6px] grid grid-cols-2 gap-[11px]">
              <Button icon={<InterpolateIcon />} onClick={doInterpolate} disabled={!trained || busy || training}>
                Interpolar
              </Button>
              <Button icon={<RefreshIcon />} onClick={shuffleInterpolate} disabled={!trained || busy || training}>
                Nueva interpolación
              </Button>
            </div>
          </Card>
        </div>

        {/* DERECHA: entrenamiento */}
        <Card rise={3}>
          <CardHeader
            title="Entrenamiento"
            actions={<Tag variant="warn">requiere reentrenar</Tag>}
            onInfo={() => open(C.topics.parametros)}
            infoLabel="Parámetros del modelo"
          />
          <Slider label="z_dim" value={zDim} min={16} max={256} step={8} onChange={setZDim} />
          <Slider
            label="learning_rate"
            value={lrIdx}
            min={0}
            max={LR_VALUES.length - 1}
            onChange={setLrIdx}
            format={() => lr}
          />
          <Slider label="epochs" value={epochs} min={1} max={100} onChange={setEpochs} />

          <div className="gm-ctrl">
            <div className="row">
              <span className="name">seed</span>
              <input
                type="number"
                className="gm-val"
                style={{ width: 72 }}
                value={trainSeed}
                onChange={(e) => setTrainSeed(Number(e.target.value))}
              />
            </div>
          </div>

          <Button
            variant="ghost"
            full
            icon={<RefreshIcon />}
            onClick={resetDefaults}
            disabled={training}
            className="mb-[11px]"
          >
            Volver a parámetros por defecto
          </Button>

          <StatusLine>{statusLine}</StatusLine>

          {/* Curvas de pérdida G y D (dos series con distinto color). */}
          {liveLoss.length > 0 ? (
            <div className="mb-3 mt-2 flex flex-col gap-3">
              <div>
                <div className="mb-1 flex items-center gap-2 text-[12px] text-[color:var(--slate)]">
                  <span className="inline-block h-2 w-3 rounded-full" style={{ background: 'var(--pink)' }} />
                  pérdida del generador (g_loss)
                </div>
                <TrainingChart points={gPoints} color="var(--pink)" />
              </div>
              <div>
                <div className="mb-1 flex items-center gap-2 text-[12px] text-[color:var(--slate)]">
                  <span className="inline-block h-2 w-3 rounded-full" style={{ background: 'var(--violet)' }} />
                  pérdida del discriminador (d_loss)
                </div>
                <TrainingChart points={dPoints} color="var(--violet)" />
              </div>
            </div>
          ) : (
            <p className="mb-3 mt-2 text-[12.5px] text-[color:var(--muted)]">
              Entrena para ver las curvas de pérdida de G y D en vivo.
            </p>
          )}

          {/* Preview del último epoch: cómo mejora el generador (z fijo). */}
          {(training || preview) && (
            <div className="mb-3">
              <div className="mb-1 text-[12px] text-[color:var(--slate)]">
                Vista previa del generador {trainInfo ? `(epoch ${trainInfo.epoch})` : ''}
              </div>
              <div className="grid grid-cols-6 gap-[6px]">
                {preview
                  ? preview.map((img, i) => (
                      <div key={i} className="gm-thumb !rounded-[10px]">
                        <img src={img} alt={`preview ${i}`} />
                      </div>
                    ))
                  : Array.from({ length: 6 }).map((_, i) => (
                      <Skeleton key={i} className="aspect-square w-full !rounded-[10px]" />
                    ))}
              </div>
            </div>
          )}

          <div className="mb-[11px]">
            <SegmentedControl
              value={mode}
              onChange={setMode}
              options={[
                { label: 'Rápido', value: 'quick' },
                { label: 'Completo', value: 'full' },
              ]}
            />
          </div>

          {training ? (
            <Button variant="primary" full onClick={onCancel}>
              Cancelar entrenamiento
            </Button>
          ) : (
            <Button variant="primary" full icon={<PlayIcon />} onClick={onTrain} disabled={busy}>
              Entrenar
            </Button>
          )}
        </Card>
      </div>
    </>
  )
}
