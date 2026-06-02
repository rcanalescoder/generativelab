import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Card, CardHeader, SegmentedControl, Skeleton, Slider, StatusLine, Tag } from '../components/ui'
import { DatasetCard } from '../components/lab/DatasetCard'
import { ModelFlowCard } from '../components/lab/ModelFlowCard'
import { LatentMap } from '../components/lab/LatentMap'
import { ResultsCard } from '../components/lab/ResultsCard'
import { TrainingChart } from '../components/lab/TrainingChart'
import { GridSearchModal } from '../components/lab/GridSearchModal'
import {
  ClustersIcon,
  GridIcon,
  InterpolateIcon,
  PlayIcon,
  RefreshIcon,
  SearchIcon,
  SparklesIcon,
} from '../lib/icons'
import {
  cancelVAEGridSearch,
  cancelVAETraining,
  computeVAEClusters,
  generateVAE,
  getVAENeighbors,
  getVAEProjection,
  getVAEStatus,
  gridSearchVAE,
  interpolateVAE,
  reconstructVAE,
  trainVAE,
  type InterpResult,
  type NeighborsResult,
  type Projection,
  type ReconItem,
  type Sample,
  type VAELossPoint,
  type VAEStatus,
} from '../lib/api-vae'
import { getDatasetInfo, getSamples, uploadImage, type DatasetInfo, type GridConfig } from '../lib/api'
import { useInfo } from '../components/info/InfoProvider'
import vaeContent from '../content/vae.json'
import type { ModelContent } from '../content/types'

const C = vaeContent as unknown as ModelContent

const LR_VALUES = [0.0001, 0.0003, 0.001, 0.003, 0.01]
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
  latentDim: 128,
  lrIdx: 2,
  epochs: 12,
  loss: 'mse' as const,
  beta: 1,
  noise: 0,
  k: 8,
  nClusters: 5,
  steps: 6,
  seed: 42,
  mode: 'quick' as const,
  earlyStop: true,
  genN: 8,
  genSeed: 42,
}

// β se maneja con un slider entero (×100) para tener pasos finos sin floats en el control.
const BETA_MIN = 0
const BETA_MAX = 400 // → β ∈ [0, 4]
const betaFromSlider = (v: number) => v / 100
const sliderFromBeta = (b: number) => Math.round(b * 100)

export function VAETab() {
  const { open } = useInfo()
  const [status, setStatus] = useState<VAEStatus | null>(null)
  const [info, setInfo] = useState<DatasetInfo | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])
  const [gallerySeed, setGallerySeed] = useState(0)
  const [pipelineId, setPipelineId] = useState<number | null>(null)
  const [pipelineItem, setPipelineItem] = useState<ReconItem | null>(null)
  const [projection, setProjection] = useState<Projection | null>(null)
  const [method, setMethod] = useState<'pca' | 'umap'>('pca')
  const [neighbors, setNeighbors] = useState<NeighborsResult | null>(null)
  const [interp, setInterp] = useState<InterpResult | null>(null)
  const [interpB, setInterpB] = useState<Sample | null>(null)
  const [alpha, setAlpha] = useState(0.5)

  // generación (muestreo del prior) — rasgo clave del VAE
  const [generated, setGenerated] = useState<string[] | null>(null)
  const [genN, setGenN] = useState(DEFAULTS.genN)
  const [genSeed, setGenSeed] = useState(DEFAULTS.genSeed)

  // hiperparámetros del modelo (requieren reentrenar)
  const [latentDim, setLatentDim] = useState(DEFAULTS.latentDim)
  const [lrIdx, setLrIdx] = useState(DEFAULTS.lrIdx)
  const [epochs, setEpochs] = useState(DEFAULTS.epochs)
  const [loss, setLoss] = useState<'mse' | 'l1'>(DEFAULTS.loss)
  const [beta, setBeta] = useState(DEFAULTS.beta)
  // exploración (interactivo)
  const [noise, setNoise] = useState(DEFAULTS.noise)
  const [k, setK] = useState(DEFAULTS.k)
  const [nClusters, setNClusters] = useState(DEFAULTS.nClusters)
  const [steps, setSteps] = useState(DEFAULTS.steps)
  const [seed, setSeed] = useState(DEFAULTS.seed)
  const [mode, setMode] = useState<'quick' | 'full'>(DEFAULTS.mode)
  const [earlyStop, setEarlyStop] = useState(DEFAULTS.earlyStop)

  const [training, setTraining] = useState(false)
  const [liveLoss, setLiveLoss] = useState<VAELossPoint[]>([])
  const [trainInfo, setTrainInfo] = useState<{ epoch: number; epochs: number; loss: number; recon: number; kl: number } | null>(null)
  const [stoppedEarly, setStoppedEarly] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [genBusy, setGenBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [gridOpen, setGridOpen] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const lr = LR_VALUES[lrIdx]
  const trained = !!status?.trained
  const outdated =
    trained &&
    !!status &&
    (latentDim !== status.hyperparams.latent_dim ||
      lr !== status.hyperparams.learning_rate ||
      epochs !== status.hyperparams.epochs ||
      loss !== status.hyperparams.loss ||
      beta !== status.hyperparams.beta)

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

  // carga inicial
  useEffect(() => {
    ;(async () => {
      try {
        const st = await getVAEStatus()
        setStatus(st)
        setLatentDim(st.hyperparams.latent_dim)
        setEpochs(st.hyperparams.epochs)
        setLoss(st.hyperparams.loss)
        setBeta(st.hyperparams.beta)
        setLrIdx(nearestLrIdx(st.hyperparams.learning_rate))
        setSeed(st.seed)
        setInfo(await getDatasetInfo())
        const sm = await getSamples(8, 0)
        setSamples(sm)
        if (st.trained && sm.length) {
          const pid = sm[0].id
          setPipelineId(pid)
          const items = await reconstructVAE([pid], 0)
          setPipelineItem(items[0] ?? null)
          setProjection(await computeVAEClusters(5))
          setNeighbors(await getVAENeighbors(pid, 8))
          const bSample = sm[Math.min(1, sm.length - 1)]
          setInterpB(bSample ?? null)
          setInterp(await interpolateVAE(pid, bSample.id, 6))
          setGenerated(await generateVAE(DEFAULTS.genN, DEFAULTS.genSeed))
        }
      } catch (e) {
        setError(errMsg(e))
      }
    })()
  }, [])

  const selectSample = (id: number) =>
    run(async () => {
      setPipelineId(id)
      const items = await reconstructVAE([id], noise)
      setPipelineItem(items[0] ?? null)
    })

  const shuffle = () =>
    run(async () => {
      const s = gallerySeed + 1
      setGallerySeed(s)
      setSamples(await getSamples(8, s))
    })

  const onUpload = (file: File) =>
    run(async () => {
      // El backend de upload reconstruye con el AE; para el VAE usamos su propio endpoint
      // a partir del id, así que aquí reutilizamos uploadImage solo como reconstrucción visual.
      setPipelineItem(await uploadImage(file))
      setPipelineId(null)
    })

  const doReconstruct = () =>
    run(async () => {
      if (pipelineId == null) return
      const items = await reconstructVAE([pipelineId], noise)
      setPipelineItem(items[0] ?? null)
    })

  const doNeighbors = () =>
    run(async () => {
      if (pipelineId == null) return
      setNeighbors(await getVAENeighbors(pipelineId, k))
    })

  const doClusters = () =>
    run(async () => {
      setProjection(await computeVAEClusters(nClusters))
    })

  const doInterpolate = () =>
    run(async () => {
      const a = pipelineId ?? samples[0]?.id ?? 0
      const b = interpB?.id ?? samples[Math.min(1, samples.length - 1)]?.id ?? a
      setInterp(await interpolateVAE(a, b, steps))
    })

  const changeB = () =>
    run(async () => {
      if (!samples.length) return
      const cands = samples.filter((s) => s.id !== pipelineId)
      const pick = cands.length ? cands[Math.floor(Math.random() * cands.length)] : samples[0]
      setInterpB(pick)
      const a = pipelineId ?? samples[0].id
      setInterp(await interpolateVAE(a, pick.id, steps))
    })

  const changeMethod = (m: 'pca' | 'umap') =>
    run(async () => {
      setMethod(m)
      setProjection(await getVAEProjection(m))
    })

  const doGenerate = async () => {
    setGenBusy(true)
    setError(null)
    try {
      setGenerated(await generateVAE(genN, genSeed))
    } catch (e) {
      setError(errMsg(e))
    } finally {
      setGenBusy(false)
    }
  }

  const onTrain = () => {
    setError(null)
    setTraining(true)
    setLiveLoss([])
    setTrainInfo(null)
    setStoppedEarly(null)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    trainVAE(
      { mode, seed, early_stop: earlyStop, hyperparams: { latent_dim: latentDim, learning_rate: lr, epochs, loss, beta } },
      (ev) => {
        if (ev.type === 'epoch') {
          setLiveLoss((prev) => [...prev, { epoch: ev.epoch, loss: ev.loss, recon: ev.recon_loss, kl: ev.kl_loss }])
          setTrainInfo({ epoch: ev.epoch, epochs: ev.epochs, loss: ev.loss, recon: ev.recon_loss, kl: ev.kl_loss })
        } else if (ev.type === 'done') {
          setLiveLoss(ev.loss_history)
          if (ev.stopped_early) setStoppedEarly(ev.epochs_run ?? ev.loss_history.length)
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
        // refrescar el estado y los paneles con el modelo recién entrenado
        run(async () => {
          const st = await getVAEStatus()
          setStatus(st)
          if (st.trained) {
            if (pipelineId != null) {
              const items = await reconstructVAE([pipelineId], noise)
              setPipelineItem(items[0] ?? null)
              setNeighbors(await getVAENeighbors(pipelineId, k))
            }
            setProjection(await computeVAEClusters(nClusters))
            setGenerated(await generateVAE(genN, genSeed))
          }
        })
      })
  }

  const onCancel = () => {
    cancelVAETraining().catch(() => undefined)
    abortRef.current?.abort()
  }

  const resetDefaults = () => {
    setLatentDim(DEFAULTS.latentDim)
    setLrIdx(DEFAULTS.lrIdx)
    setEpochs(DEFAULTS.epochs)
    setLoss(DEFAULTS.loss)
    setBeta(DEFAULTS.beta)
    setNoise(DEFAULTS.noise)
    setK(DEFAULTS.k)
    setNClusters(DEFAULTS.nClusters)
    setSteps(DEFAULTS.steps)
    setSeed(DEFAULTS.seed)
    setMode(DEFAULTS.mode)
    setEarlyStop(DEFAULTS.earlyStop)
  }

  const applyGrid = (cfg: GridConfig) => {
    setLatentDim(cfg.latent_dim)
    setLrIdx(nearestLrIdx(cfg.learning_rate))
    setLoss(cfg.loss)
    setEpochs(cfg.epochs)
  }

  const statusLine = training ? (
    trainInfo ? (
      <>
        epoch <b>{trainInfo.epoch}/{trainInfo.epochs}</b> · loss{' '}
        <span className="mono">{trainInfo.loss.toFixed(4)}</span> · recon{' '}
        <span className="mono">{trainInfo.recon.toFixed(4)}</span> · KL{' '}
        <span className="mono">{trainInfo.kl.toFixed(4)}</span>
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
      {stoppedEarly != null && (
        <span style={{ color: '#0a6b4e' }}> · parado temprano (epoch {stoppedEarly}, sin mejora)</span>
      )}
    </>
  ) : (
    <>sin entrenar — entrena o carga el checkpoint demo</>
  )

  // Curvas: total/recon (azul/violeta) y KL (ámbar). TrainingChart toma {epoch, loss}.
  const totalPts = liveLoss.map((p) => ({ epoch: p.epoch, loss: p.loss }))
  const reconPts = liveLoss.map((p) => ({ epoch: p.epoch, loss: p.recon }))
  const klPts = liveLoss.map((p) => ({ epoch: p.epoch, loss: p.kl }))

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

      <div className="mb-[22px] grid items-stretch gap-[22px] grid-cols-1 xl:grid-cols-[minmax(316px,358px)_minmax(0,1fr)_minmax(336px,374px)]">
        {/* IZQUIERDA: dataset + exploración */}
        <div className="flex flex-col gap-[22px]">
          <DatasetCard
            info={info}
            samples={samples}
            onShuffle={shuffle}
            onUpload={onUpload}
            onSelect={selectSample}
            onInfo={() => open(C.topics.dataset)}
            busy={busy}
          />

          <Card rise={2}>
            <CardHeader
              title="Exploración"
              actions={<Tag variant="live">interactivo</Tag>}
              onInfo={() => open(C.topics['parametros-exploracion'])}
              infoLabel="Parámetros de exploración"
            />
            <Slider
              label="ruido en z"
              value={Math.round(noise * 100)}
              min={0}
              max={100}
              onChange={(v) => setNoise(v / 100)}
              format={(v) => (v / 100).toFixed(2)}
            />
            <Slider label="k vecinos" value={k} min={1} max={24} onChange={setK} />
            <Slider label="n clusters" value={nClusters} min={2} max={12} onChange={setNClusters} />
            <Slider label="pasos interpolación" value={steps} min={2} max={12} onChange={setSteps} />

            <div className="mt-[8px] grid grid-cols-2 gap-[11px]">
              <Button icon={<RefreshIcon />} onClick={doReconstruct} disabled={!trained || busy || training}>
                Reconstruir
              </Button>
              <Button icon={<SearchIcon />} onClick={doNeighbors} disabled={!trained || busy || training}>
                Buscar similares
              </Button>
              <Button
                className="col-span-2"
                icon={<InterpolateIcon />}
                onClick={doInterpolate}
                disabled={!trained || busy || training}
              >
                Interpolar
              </Button>
              <Button
                className="col-span-2"
                icon={<ClustersIcon />}
                onClick={doClusters}
                disabled={!trained || busy || training}
              >
                Calcular clusters
              </Button>
            </div>
          </Card>
        </div>

        {/* CENTRO: flujo + mapa + generación */}
        <div className="flex min-w-0 flex-col gap-[22px]">
          <ModelFlowCard item={pipelineItem} latentDim={latentDim} onInfo={() => open(C.topics.flujo)} />
          <LatentMap
            projection={projection}
            method={method}
            onMethodChange={changeMethod}
            loading={busy && !projection}
            onInfo={() => open(C.topics['mapa-latente'])}
          />

          {/* GENERACIÓN: el rasgo distintivo del VAE */}
          <Card rise={4}>
            <CardHeader
              title="Generación (muestrear del prior)"
              actions={<Tag variant="live">z ~ N(0, I)</Tag>}
              onInfo={() => open(C.topics.generacion)}
              infoLabel="Cómo genera caras un VAE"
            />
            <div className="mb-[12px] text-[13.5px] leading-[1.5] text-[color:var(--slate)]">
              Tomamos vectores <span className="mono">z</span> al azar del prior{' '}
              <span className="mono">N(0,&nbsp;I)</span> y los pasamos por el decoder. No hay imagen de
              entrada: cada cara es nueva. Es lo que un autoencoder normal no puede hacer.
            </div>

            <div className="mb-[12px] grid grid-cols-1 gap-x-[16px] sm:grid-cols-2">
              <Slider label="nº de muestras" value={genN} min={4} max={24} onChange={setGenN} />
              <div className="gm-ctrl">
                <div className="row">
                  <span className="name">semilla</span>
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

            <div
              className="grid gap-[10px]"
              style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(72px, 1fr))' }}
            >
              {generated
                ? generated.map((img, i) => (
                    <div key={i} className="gm-thumb overflow-hidden rounded-[12px]">
                      <img src={img} alt={`generada ${i + 1}`} loading="lazy" />
                    </div>
                  ))
                : Array.from({ length: genN }).map((_, i) => (
                    <Skeleton key={i} className="aspect-square rounded-[12px]" />
                  ))}
            </div>

            <Button
              variant="primary"
              full
              icon={<SparklesIcon />}
              onClick={doGenerate}
              disabled={!trained || genBusy || training}
              className="mt-[14px]"
            >
              {genBusy ? 'Generando…' : 'Generar nuevas'}
            </Button>
          </Card>
        </div>

        {/* DERECHA: entrenamiento */}
        <Card rise={3}>
          <CardHeader
            title="Entrenamiento"
            actions={<Tag variant="warn">requiere reentrenar</Tag>}
            onInfo={() => open(C.topics['parametros-modelo'])}
            infoLabel="Parámetros del modelo"
          />
          <Slider label="latent_dim" value={latentDim} min={16} max={256} step={8} onChange={setLatentDim} />
          <Slider label="learning_rate" value={lrIdx} min={0} max={LR_VALUES.length - 1} onChange={setLrIdx} format={() => lr} />
          <Slider label="epochs" value={epochs} min={1} max={100} onChange={setEpochs} />
          <div className="gm-ctrl">
            <div className="row" style={{ marginBottom: 0 }}>
              <span className="name">loss</span>
              <SegmentedControl
                value={loss}
                onChange={setLoss}
                options={[
                  { label: 'MSE', value: 'mse' },
                  { label: 'L1', value: 'l1' },
                ]}
              />
            </div>
          </div>
          <Slider
            label="β (peso del KL)"
            value={sliderFromBeta(beta)}
            min={BETA_MIN}
            max={BETA_MAX}
            step={5}
            onChange={(v) => setBeta(betaFromSlider(v))}
            format={(v) => betaFromSlider(v).toFixed(2)}
          />

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
            <div className="mb-3 flex flex-col gap-2">
              <div>
                <div className="mb-0.5 flex items-center gap-3 text-[11.5px] text-[color:var(--muted)]">
                  <span className="flex items-center gap-1">
                    <span className="inline-block h-[3px] w-[12px] rounded-full" style={{ background: 'var(--blue)' }} />
                    total
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="inline-block h-[3px] w-[12px] rounded-full" style={{ background: 'var(--violet)' }} />
                    reconstrucción
                  </span>
                </div>
                <div className="relative">
                  <TrainingChart points={totalPts} color="var(--blue)" />
                  {/* curva de reconstrucción superpuesta (mismo eje) */}
                  <div className="pointer-events-none absolute inset-0">
                    <TrainingChart points={reconPts} color="var(--violet)" />
                  </div>
                </div>
              </div>
              <div>
                <div className="mb-0.5 flex items-center gap-1 text-[11.5px] text-[color:var(--muted)]">
                  <span className="inline-block h-[3px] w-[12px] rounded-full" style={{ background: 'var(--amber)' }} />
                  KL (regularización hacia N(0, I))
                </div>
                <TrainingChart points={klPts} color="var(--amber)" />
              </div>
            </div>
          )}

          <div className="gm-ctrl">
            <div className="row" style={{ marginBottom: 0 }}>
              <span className="name">early stop</span>
              <SegmentedControl
                value={earlyStop ? 'on' : 'off'}
                onChange={(v) => setEarlyStop(v === 'on')}
                options={[
                  { label: 'Sí', value: 'on' },
                  { label: 'No', value: 'off' },
                ]}
              />
            </div>
          </div>

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

          <Button
            full
            icon={<GridIcon />}
            onClick={() => setGridOpen(true)}
            disabled={busy || training}
            className="mt-[11px]"
          >
            Buscar mejores parámetros
          </Button>
        </Card>
      </div>

      <ResultsCard
        neighbors={neighbors}
        interp={interp}
        alpha={alpha}
        onAlpha={setAlpha}
        aImage={pipelineItem?.original}
        bImage={interpB?.image}
        onChangeB={changeB}
        busy={busy}
        onInfo={() => open(C.topics.resultados)}
      />

      {gridOpen && (
        <GridSearchModal
          seed={seed}
          onClose={() => setGridOpen(false)}
          onApply={applyGrid}
          runSearch={gridSearchVAE}
          cancelSearch={cancelVAEGridSearch}
          title="Buscar mejores parámetros (VAE)"
        />
      )}
    </>
  )
}
