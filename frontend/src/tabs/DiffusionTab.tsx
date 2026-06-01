import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Card, CardHeader, SegmentedControl, Skeleton, Slider, StatusLine, Tag } from '../components/ui'
import { TrainingChart } from '../components/lab/TrainingChart'
import { PlayIcon, RefreshIcon, SparklesIcon } from '../lib/icons'
import {
  cancelDiffusionTraining,
  generateDiffusion,
  getDiffusionStatus,
  getTrajectory,
  trainDiffusion,
  type DiffusionStatus,
  type TrajectoryFrame,
} from '../lib/api-diffusion'
import type { LossPoint } from '../lib/api'
import { useInfo } from '../components/info/InfoProvider'
import diffusionContent from '../content/diffusion.json'
import type { ModelContent } from '../content/types'

const C = diffusionContent as unknown as ModelContent

const LR_VALUES = [0.00005, 0.0001, 0.0002, 0.0005, 0.001]
const nearestLrIdx = (lr: number) => {
  let best = 0
  for (let i = 1; i < LR_VALUES.length; i++) {
    if (Math.abs(LR_VALUES[i] - lr) < Math.abs(LR_VALUES[best] - lr)) best = i
  }
  return best
}
const errMsg = (e: unknown) => (e instanceof Error ? e.message : String(e))

/** Valores por defecto (los que ves al abrir la app, = checkpoint demo). */
const DEFAULTS = {
  lrIdx: 2, // 0.0002
  epochs: 18,
  seed: 42,
  mode: 'quick' as const,
  genN: 6,
  genSeed: 42,
  genSteps: 50,
  trajSeed: 7,
  trajSteps: 50,
  snapshots: 8,
}

export function DiffusionTab() {
  const { open } = useInfo()
  const [status, setStatus] = useState<DiffusionStatus | null>(null)

  // galería generada
  const [generated, setGenerated] = useState<string[] | null>(null)
  const [genN, setGenN] = useState(DEFAULTS.genN)
  const [genSeed, setGenSeed] = useState(DEFAULTS.genSeed)
  const [genSteps, setGenSteps] = useState(DEFAULTS.genSteps)
  const [genBusy, setGenBusy] = useState(false)

  // proceso de difusión (trayectoria)
  const [frames, setFrames] = useState<TrajectoryFrame[] | null>(null)
  const [trajSeed, setTrajSeed] = useState(DEFAULTS.trajSeed)
  const [trajSteps, setTrajSteps] = useState(DEFAULTS.trajSteps)
  const [snapshots, setSnapshots] = useState(DEFAULTS.snapshots)
  const [trajBusy, setTrajBusy] = useState(false)

  // hiperparámetros del modelo (requieren reentrenar)
  const [lrIdx, setLrIdx] = useState(DEFAULTS.lrIdx)
  const [epochs, setEpochs] = useState(DEFAULTS.epochs)
  const [seed, setSeed] = useState(DEFAULTS.seed)
  const [mode, setMode] = useState<'quick' | 'full'>(DEFAULTS.mode)

  const [training, setTraining] = useState(false)
  const [liveLoss, setLiveLoss] = useState<LossPoint[]>([])
  const [trainInfo, setTrainInfo] = useState<{ epoch: number; epochs: number; loss: number } | null>(null)
  const [preview, setPreview] = useState<string[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const lr = LR_VALUES[lrIdx]
  const trained = !!status?.trained
  const busy = genBusy || trajBusy
  const outdated =
    trained &&
    !!status &&
    (lr !== status.hyperparams.learning_rate || epochs !== status.hyperparams.epochs)

  // ---------- acciones ----------
  const doGenerate = useCallback(
    async (n = genN, sd = genSeed, st = genSteps) => {
      setGenBusy(true)
      setError(null)
      try {
        setGenerated(await generateDiffusion(n, sd, st))
      } catch (e) {
        setError(errMsg(e))
      } finally {
        setGenBusy(false)
      }
    },
    [genN, genSeed, genSteps],
  )

  const doTrajectory = useCallback(
    async (sd = trajSeed, st = trajSteps, sn = snapshots) => {
      setTrajBusy(true)
      setError(null)
      try {
        setFrames(await getTrajectory(sd, st, sn))
      } catch (e) {
        setError(errMsg(e))
      } finally {
        setTrajBusy(false)
      }
    },
    [trajSeed, trajSteps, snapshots],
  )

  // carga inicial: status y, si hay modelo, una generación y una trayectoria.
  // Las llamadas iniciales van en línea (con los valores por defecto) para no depender de
  // los callbacks y mantener el efecto de montaje sin dependencias.
  useEffect(() => {
    ;(async () => {
      try {
        const st = await getDiffusionStatus()
        setStatus(st)
        setEpochs(st.hyperparams.epochs)
        setLrIdx(nearestLrIdx(st.hyperparams.learning_rate))
        setSeed(st.seed)
        if (st.trained) {
          setGenerated(await generateDiffusion(DEFAULTS.genN, DEFAULTS.genSeed, DEFAULTS.genSteps))
          setFrames(await getTrajectory(DEFAULTS.trajSeed, DEFAULTS.trajSteps, DEFAULTS.snapshots))
        }
      } catch (e) {
        setError(errMsg(e))
      }
    })()
  }, [])

  const newGenSeed = () => {
    const s = genSeed + 1
    setGenSeed(s)
    void doGenerate(genN, s, genSteps)
  }

  const newTrajSeed = () => {
    const s = trajSeed + 1
    setTrajSeed(s)
    void doTrajectory(s, trajSteps, snapshots)
  }

  const onTrain = () => {
    setError(null)
    setTraining(true)
    setLiveLoss([])
    setTrainInfo(null)
    setPreview(null)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    trainDiffusion(
      { mode, seed, hyperparams: { learning_rate: lr, epochs } },
      (ev) => {
        if (ev.type === 'epoch') {
          setLiveLoss((prev) => [...prev, { epoch: ev.epoch, loss: ev.loss }])
          setTrainInfo({ epoch: ev.epoch, epochs: ev.epochs, loss: ev.loss })
          setPreview(ev.preview)
        } else if (ev.type === 'step') {
          setTrainInfo((prev) =>
            prev ? { ...prev, loss: ev.loss } : { epoch: 1, epochs, loss: ev.loss },
          )
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
        // refrescar estado + paneles con el modelo recién entrenado
        ;(async () => {
          try {
            const st = await getDiffusionStatus()
            setStatus(st)
            if (st.trained) {
              await doGenerate(genN, genSeed, genSteps)
              await doTrajectory(trajSeed, trajSteps, snapshots)
            }
          } catch (e) {
            setError(errMsg(e))
          }
        })()
      })
  }

  const onCancel = () => {
    cancelDiffusionTraining().catch(() => undefined)
    abortRef.current?.abort()
  }

  const resetDefaults = () => {
    setLrIdx(DEFAULTS.lrIdx)
    setEpochs(DEFAULTS.epochs)
    setSeed(DEFAULTS.seed)
    setMode(DEFAULTS.mode)
  }

  const statusLine = training ? (
    trainInfo ? (
      <>
        epoch <b>{trainInfo.epoch}/{trainInfo.epochs}</b> · loss{' '}
        <span className="mono">{trainInfo.loss.toFixed(4)}</span>
      </>
    ) : (
      <>iniciando entrenamiento…</>
    )
  ) : outdated ? (
    <span style={{ color: '#B5740B' }}>cambios sin aplicar — reentrena para verlos</span>
  ) : trained && status?.loss_history.length ? (
    <>
      modelo entrenado · loss final{' '}
      <span className="mono">{status.loss_history[status.loss_history.length - 1].loss.toFixed(4)}</span>
    </>
  ) : (
    <>sin entrenar — entrena o genera el checkpoint demo</>
  )

  const emptyHint = !trained && !training

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

      <div className="mb-[22px] grid items-start gap-[22px] grid-cols-1 xl:grid-cols-[minmax(0,1fr)_minmax(336px,374px)]">
        {/* IZQUIERDA / CENTRO: galería + proceso */}
        <div className="flex min-w-0 flex-col gap-[22px]">
          {/* --- Galería generada --- */}
          <Card rise={1}>
            <CardHeader
              title="Galería generada"
              actions={<Tag variant="live">desde ruido</Tag>}
              onInfo={() => open(C.topics.generacion)}
              infoLabel="Cómo se generan las caras"
            />
            <div className="gm-sub-d mb-[14px]">
              Cada cara nace de ruido aleatorio puro y se «limpia» paso a paso hasta convertirse en un
              rostro nuevo. Ninguna existe en el dataset.
            </div>

            <div
              className="grid gap-[10px]"
              style={{ gridTemplateColumns: `repeat(${Math.min(genN, 6)}, minmax(0, 1fr))` }}
            >
              {generated
                ? generated.map((src, i) => (
                    <div className="gm-thumb" key={i}>
                      <img src={src} alt={`muestra ${i + 1}`} />
                    </div>
                  ))
                : Array.from({ length: emptyHint ? 0 : genN }).map((_, i) => (
                    <Skeleton key={i} className="aspect-square w-full !rounded-[12px]" />
                  ))}
            </div>

            {emptyHint && !generated && (
              <div className="gm-caption">
                Entrena el modelo o genera el checkpoint demo para muestrear caras.
              </div>
            )}

            <div className="mt-[16px] grid grid-cols-1 gap-x-[18px] gap-y-[2px] sm:grid-cols-3">
              <Slider label="nº muestras" value={genN} min={1} max={6} onChange={setGenN} />
              <Slider
                label="pasos de muestreo"
                value={genSteps}
                min={5}
                max={status?.timesteps ?? 200}
                step={5}
                onChange={setGenSteps}
              />
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

            <Button
              variant="primary"
              full
              icon={<SparklesIcon />}
              onClick={() => doGenerate()}
              disabled={!trained || busy || training}
              className="mt-[10px]"
            >
              {genBusy ? 'Generando…' : 'Generar nuevas'}
            </Button>
            <Button
              full
              icon={<RefreshIcon />}
              onClick={newGenSeed}
              disabled={!trained || busy || training}
              className="mt-[10px]"
            >
              Nueva semilla
            </Button>
          </Card>

          {/* --- Proceso de difusión (panel estrella) --- */}
          <Card rise={2}>
            <CardHeader
              title="Proceso de difusión"
              actions={<Tag variant="live">la trayectoria</Tag>}
              onInfo={() => open(C.topics.proceso)}
              infoLabel="El proceso de difusión paso a paso"
            />
            <div className="gm-sub-d mb-[16px]">
              La misma imagen a lo largo del muestreo: de <b>ruido puro</b> (izquierda) a{' '}
              <b>cara limpia</b> (derecha). Cada instantánea quita un poco más de ruido.
            </div>

            <div
              className="grid gap-[12px]"
              style={{
                gridTemplateColumns: `repeat(${frames ? frames.length : snapshots}, minmax(0, 1fr))`,
              }}
            >
              {frames
                ? frames.map((f, i) => {
                    const isFirst = i === 0
                    const isLast = i === frames.length - 1
                    const label = isFirst ? 'ruido' : isLast ? 'cara' : `t ${f.t}`
                    return (
                      <div key={i} className="flex flex-col items-center">
                        <div
                          className="gm-pic"
                          style={{
                            width: '100%',
                            height: 'auto',
                            aspectRatio: '1',
                            boxShadow: isLast ? '0 0 0 2.5px var(--violet)' : undefined,
                          }}
                        >
                          <img src={f.image} alt={label} style={{ width: '100%', height: '100%' }} />
                        </div>
                        <div className="cap" style={{ marginTop: 8 }}>
                          {label}
                          <br />
                          <span className="mono">{isFirst ? `t=${f.t}` : isLast ? 't=0' : `${f.position}/${f.total - 1}`}</span>
                        </div>
                      </div>
                    )
                  })
                : Array.from({ length: emptyHint ? 0 : snapshots }).map((_, i) => (
                    <div key={i} className="flex flex-col items-center">
                      <Skeleton className="w-full !rounded-[15px]" style={{ aspectRatio: '1' }} />
                      <span className="cap mono" style={{ marginTop: 8 }}>
                        ·
                      </span>
                    </div>
                  ))}
            </div>

            {emptyHint && !frames && (
              <div className="gm-caption">
                Aquí verás cómo el ruido se transforma en una cara, paso a paso, en cuanto haya un modelo.
              </div>
            )}

            <div className="mt-[16px] grid grid-cols-1 gap-x-[18px] gap-y-[2px] sm:grid-cols-3">
              <Slider label="instantáneas" value={snapshots} min={3} max={12} onChange={setSnapshots} />
              <Slider
                label="pasos de muestreo"
                value={trajSteps}
                min={5}
                max={status?.timesteps ?? 200}
                step={5}
                onChange={setTrajSteps}
              />
              <div className="gm-ctrl">
                <div className="row">
                  <span className="name">seed</span>
                  <input
                    type="number"
                    className="gm-val"
                    style={{ width: 72 }}
                    value={trajSeed}
                    onChange={(e) => setTrajSeed(Number(e.target.value))}
                  />
                </div>
              </div>
            </div>

            <Button
              variant="primary"
              full
              icon={<PlayIcon />}
              onClick={() => doTrajectory()}
              disabled={!trained || busy || training}
              className="mt-[10px]"
            >
              {trajBusy ? 'Generando proceso…' : 'Nuevo proceso'}
            </Button>
            <Button
              full
              icon={<RefreshIcon />}
              onClick={newTrajSeed}
              disabled={!trained || busy || training}
              className="mt-[10px]"
            >
              Nueva semilla
            </Button>
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
          <div className="gm-sub-d mb-[12px]">
            El modelo aprende a predecir el ruido que se añadió a una imagen. La pérdida es el error de
            esa predicción (MSE).
          </div>

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
            <div className="row" style={{ marginBottom: 0 }}>
              <span className="name">timesteps (T)</span>
              <span className="gm-val">{status?.timesteps ?? '—'}</span>
            </div>
          </div>

          <div className="gm-ctrl">
            <div className="row">
              <span className="name">seed</span>
              <input
                type="number"
                className="gm-val"
                style={{ width: 72 }}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
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

          {liveLoss.length > 0 && (
            <TrainingChart points={liveLoss} className="mb-3" color="var(--violet)" />
          )}

          {/* vista previa del último epoch durante el entrenamiento */}
          {training && preview && preview.length > 0 && (
            <div className="mb-3">
              <div className="gm-sub-h" style={{ marginBottom: 8 }}>
                Vista previa (último epoch)
              </div>
              <div
                className="grid gap-[8px]"
                style={{ gridTemplateColumns: `repeat(${Math.min(preview.length, 6)}, minmax(0, 1fr))` }}
              >
                {preview.map((src, i) => (
                  <div className="gm-thumb" key={i}>
                    <img src={src} alt={`preview ${i + 1}`} />
                  </div>
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

          <div className="gm-caption">
            Entrenar no toca el checkpoint demo: el modelo entrenado vive durante esta sesión. La
            difusión es costosa, así que «Rápido» usa un subconjunto y pocas épocas.
          </div>
        </Card>
      </div>
    </>
  )
}
