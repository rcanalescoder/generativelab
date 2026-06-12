import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Card, CardHeader, SegmentedControl, Slider, StatusLine, Tag, Hint } from '../components/ui'
import { DatasetCard } from '../components/lab/DatasetCard'
import { ModelFlowCard } from '../components/lab/ModelFlowCard'
import { LatentMap } from '../components/lab/LatentMap'
import { ResultsCard } from '../components/lab/ResultsCard'
import { TrainingChart } from '../components/lab/TrainingChart'
import { ClustersIcon, GridIcon, InterpolateIcon, LayersIcon, PlayIcon, RefreshIcon, SearchIcon, SparklesIcon } from '../lib/icons'
import {
  cancelTraining,
  computeClusters,
  getAEStatus,
  getDatasetInfo,
  getNeighbors,
  getProjection,
  getSamples,
  interpolate,
  reconstruct,
  setAEArch,
  trainAE,
  uploadImage,
  type AEArch,
  type AEStatus,
  type DatasetInfo,
  type GridConfig,
  type InterpResult,
  type LossPoint,
  type NeighborsResult,
  type Projection,
  type ReconItem,
  type Sample,
} from '../lib/api'
import { useInfo } from '../components/info/InfoProvider'
import { autoencoderContent } from '../content'
import type { ModelContent } from '../content/types'
import { GridSearchModal } from '../components/lab/GridSearchModal'
import { ArchitectureModal } from '../components/lab/ArchitectureModal'
import { MetricsPanel } from '../components/lab/MetricsPanel'
import { aeArchitecture } from '../content/architectures'

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
  arch: 'basico' as const,
  latentDim: 128,
  lrIdx: 2,
  epochs: 10,
  loss: 'mse' as const,
  noise: 0,
  k: 8,
  nClusters: 5,
  steps: 6,
  seed: 42,
  mode: 'quick' as const,
  earlyStop: true,
}

export function AutoencoderTab() {
  const { open } = useInfo()
  const C = autoencoderContent
  const H = (C.hints ?? {}) as NonNullable<ModelContent['hints']>
  const [status, setStatus] = useState<AEStatus | null>(null)
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

  // hiperparámetros del modelo (requieren reentrenar)
  const [arch, setArch] = useState<AEArch>('basico')
  const [latentDim, setLatentDim] = useState(128)
  const [lrIdx, setLrIdx] = useState(2)
  const [epochs, setEpochs] = useState(10)
  const [loss, setLoss] = useState<'mse' | 'l1'>('mse')
  // exploración (interactivo)
  const [noise, setNoise] = useState(0)
  const [k, setK] = useState(8)
  const [nClusters, setNClusters] = useState(5)
  const [steps, setSteps] = useState(6)
  const [seed, setSeed] = useState(42)
  const [mode, setMode] = useState<'quick' | 'full'>('quick')
  const [earlyStop, setEarlyStop] = useState(true)

  const [training, setTraining] = useState(false)
  const [liveLoss, setLiveLoss] = useState<LossPoint[]>([])
  const [trainInfo, setTrainInfo] = useState<{ epoch: number; epochs: number; loss: number } | null>(null)
  const [stoppedEarly, setStoppedEarly] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [gridOpen, setGridOpen] = useState(false)
  const [archOpen, setArchOpen] = useState(false)
  const [metricsOpen, setMetricsOpen] = useState(false)
  // true cuando la variante elegida aún no tiene checkpoint demo (hay que entrenarla)
  const [archMissing, setArchMissing] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const lr = LR_VALUES[lrIdx]
  const trained = !!status?.trained
  const outdated =
    trained &&
    !!status &&
    // el backend ya lo marca (p. ej. al elegir una variante sin demo entrenado)…
    (status.outdated ||
      // …o el usuario tocó un parámetro del modelo respecto al checkpoint cargado.
      arch !== status.arch ||
      latentDim !== status.hyperparams.latent_dim ||
      lr !== status.hyperparams.learning_rate ||
      epochs !== status.hyperparams.epochs ||
      loss !== status.hyperparams.loss)

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
        const st = await getAEStatus()
        setStatus(st)
        setArch(st.arch)
        setLatentDim(st.hyperparams.latent_dim)
        setEpochs(st.hyperparams.epochs)
        setLoss(st.hyperparams.loss)
        setLrIdx(nearestLrIdx(st.hyperparams.learning_rate))
        setSeed(st.seed)
        setInfo(await getDatasetInfo())
        const sm = await getSamples(8, 0)
        setSamples(sm)
        if (st.trained && sm.length) {
          const pid = sm[0].id
          setPipelineId(pid)
          const items = await reconstruct([pid], 0)
          setPipelineItem(items[0] ?? null)
          setProjection(await computeClusters(5))
          setNeighbors(await getNeighbors(pid, 8))
          const bSample = sm[Math.min(1, sm.length - 1)]
          setInterpB(bSample ?? null)
          setInterp(await interpolate(pid, bSample.id, 6))
        }
      } catch (e) {
        setError(errMsg(e))
      }
    })()
  }, [])

  /** Cambia de variante de arquitectura (parámetro del modelo: afecta diagrama y requiere reentrenar).
   *  Si la variante ya tiene checkpoint demo (loaded=true) se ve al instante; si no, los paneles
   *  quedan en "entrena para ver esta variante". */
  const changeArch = (next: AEArch) => {
    if (next === arch) return
    setArch(next)
    run(async () => {
      const st = await setAEArch(next)
      setStatus(st)
      setArchMissing(!st.loaded)
      if (st.loaded && st.trained) {
        // la variante ya entrenada se ve al instante: refrescamos los paneles
        const id = pipelineId ?? samples[0]?.id ?? null
        if (id != null) {
          setPipelineId(id)
          const items = await reconstruct([id], noise)
          setPipelineItem(items[0] ?? null)
          setNeighbors(await getNeighbors(id, k))
        }
        setProjection(await computeClusters(nClusters))
      } else {
        // sin demo aún: limpiamos los paneles para mostrar el estado vacío de esta variante
        setPipelineItem(null)
        setProjection(null)
        setNeighbors(null)
        setInterp(null)
      }
    })
  }

  const selectSample = (id: number) =>
    run(async () => {
      setPipelineId(id)
      const items = await reconstruct([id], noise)
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
      setPipelineItem(await uploadImage(file))
      setPipelineId(null)
    })

  const doReconstruct = () =>
    run(async () => {
      if (pipelineId == null) return
      const items = await reconstruct([pipelineId], noise)
      setPipelineItem(items[0] ?? null)
    })

  const doNeighbors = () =>
    run(async () => {
      if (pipelineId == null) return
      setNeighbors(await getNeighbors(pipelineId, k))
    })

  const doClusters = () =>
    run(async () => {
      setProjection(await computeClusters(nClusters))
    })

  const doInterpolate = () =>
    run(async () => {
      const a = pipelineId ?? samples[0]?.id ?? 0
      const b = interpB?.id ?? samples[Math.min(1, samples.length - 1)]?.id ?? a
      setInterp(await interpolate(a, b, steps))
    })

  const changeB = () =>
    run(async () => {
      if (!samples.length) return
      const cands = samples.filter((s) => s.id !== pipelineId)
      const pick = cands.length ? cands[Math.floor(Math.random() * cands.length)] : samples[0]
      setInterpB(pick)
      const a = pipelineId ?? samples[0].id
      setInterp(await interpolate(a, pick.id, steps))
    })

  const changeMethod = (m: 'pca' | 'umap') =>
    run(async () => {
      setMethod(m)
      setProjection(await getProjection(m))
    })

  const onTrain = () => {
    setError(null)
    setTraining(true)
    setLiveLoss([])
    setTrainInfo(null)
    setStoppedEarly(null)
    const ctrl = new AbortController()
    abortRef.current = ctrl
    trainAE(
      { mode, seed, early_stop: earlyStop, hyperparams: { arch, latent_dim: latentDim, learning_rate: lr, epochs, loss } },
      (ev) => {
        if (ev.type === 'epoch') {
          setLiveLoss((prev) => [...prev, { epoch: ev.epoch, loss: ev.loss }])
          setTrainInfo({ epoch: ev.epoch, epochs: ev.epochs, loss: ev.loss })
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
          const st = await getAEStatus()
          setStatus(st)
          if (st.trained) setArchMissing(false)
          if (st.trained && pipelineId != null) {
            const items = await reconstruct([pipelineId], noise)
            setPipelineItem(items[0] ?? null)
            setProjection(await computeClusters(nClusters))
            setNeighbors(await getNeighbors(pipelineId, k))
          }
        })
      })
  }

  const onCancel = () => {
    cancelTraining().catch(() => undefined)
    abortRef.current?.abort()
  }

  const resetDefaults = () => {
    if (arch !== DEFAULTS.arch) changeArch(DEFAULTS.arch)
    setLatentDim(DEFAULTS.latentDim)
    setLrIdx(DEFAULTS.lrIdx)
    setEpochs(DEFAULTS.epochs)
    setLoss(DEFAULTS.loss)
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
        <span className="mono">{trainInfo.loss.toFixed(4)}</span>
      </>
    ) : (
      <>iniciando entrenamiento…</>
    )
  ) : archMissing ? (
    <span style={{ color: '#B5740B' }}>esta variante aún no tiene demo — entrena para verla</span>
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
              hint={H.noise}
              value={Math.round(noise * 100)}
              min={0}
              max={100}
              onChange={(v) => setNoise(v / 100)}
              format={(v) => (v / 100).toFixed(2)}
            />
            <Slider hint={H.k} label="k vecinos" value={k} min={1} max={24} onChange={setK} />
            <Slider hint={H.clusters} label="n clusters" value={nClusters} min={2} max={12} onChange={setNClusters} />
            <Slider hint={H.steps_interp} label="pasos interpolación" value={steps} min={2} max={12} onChange={setSteps} />

            <div className="mt-[8px] grid grid-cols-2 gap-[11px]">
              <Button icon={<RefreshIcon />} hint={H.reconstruct} onClick={doReconstruct} disabled={!trained || busy || training}>
                Reconstruir
              </Button>
              <Button icon={<SearchIcon />} hint={H.neighbors} onClick={doNeighbors} disabled={!trained || busy || training}>
                Buscar similares
              </Button>
              <Button
                className="col-span-2"
                icon={<InterpolateIcon />}
                hint={H.interpolate_btn} onClick={doInterpolate}
                disabled={!trained || busy || training}
              >
                Interpolar
              </Button>
              <Button
                className="col-span-2"
                icon={<ClustersIcon />}
                hint={H.clusters_btn} onClick={doClusters}
                disabled={!trained || busy || training}
              >
                Calcular clusters
              </Button>
            </div>
          </Card>
        </div>

        {/* CENTRO: flujo + mapa */}
        <div className="flex min-w-0 flex-col gap-[22px]">
          <ModelFlowCard item={pipelineItem} latentDim={latentDim} onInfo={() => open(C.topics.flujo)} />
          <LatentMap
            projection={projection}
            method={method}
            onMethodChange={changeMethod}
            loading={busy && !projection}
            onInfo={() => open(C.topics['mapa-latente'])}
          />
        </div>

        {/* DERECHA: entrenamiento */}
        <Card rise={3}>
          <CardHeader
            title="Entrenamiento"
            actions={<Tag variant="warn">requiere reentrenar</Tag>}
            onInfo={() => open(C.topics['parametros-modelo'])}
            infoLabel="Parámetros del modelo"
          />
          <div className="gm-ctrl">
            <div className="row">
              <span className="name">arquitectura <Hint hint={H.arch} /></span>
            </div>
            <SegmentedControl
              value={arch}
              onChange={changeArch}
              options={[
                { label: 'Básico', value: 'basico' },
                { label: 'Grande', value: 'grande' },
                { label: 'U-Net', value: 'unet' },
              ]}
            />
          </div>
          <Slider hint={H.latent_dim} label="latent_dim" value={latentDim} min={16} max={256} step={8} onChange={setLatentDim} />
          <Slider hint={H.learning_rate} label="learning_rate" value={lrIdx} min={0} max={LR_VALUES.length - 1} onChange={setLrIdx} format={() => lr} />
          <Slider hint={H.epochs} label="epochs" value={epochs} min={1} max={100} onChange={setEpochs} />
          <div className="gm-ctrl">
            <div className="row" style={{ marginBottom: 0 }}>
              <span className="name">loss <Hint hint={H.loss} /></span>
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

          <div className="gm-ctrl">
            <div className="row">
              <span className="name">seed <Hint hint={H.seed_train} /></span>
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
            hint={H.reset} onClick={resetDefaults}
            disabled={training}
            className="mb-[11px]"
          >
            Volver a parámetros por defecto
          </Button>

          <StatusLine>{statusLine}</StatusLine>

          {liveLoss.length > 0 && <TrainingChart points={liveLoss} className="mb-3" />}

          <div className="gm-ctrl">
            <div className="row" style={{ marginBottom: 0 }}>
              <span className="name">early stop <Hint hint={H.early_stop} /></span>
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
            <div className="row">
              <span className="name">
                modo de entrenamiento
                <Hint hint={H.mode} />
              </span>
            </div>
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
            <Button variant="primary" full hint={H.cancel} onClick={onCancel}>
              Cancelar entrenamiento
            </Button>
          ) : (
            <Button variant="primary" full icon={<PlayIcon />} hint={H.train} onClick={onTrain} disabled={busy}>
              Entrenar
            </Button>
          )}

          <Button
            full
            icon={<GridIcon />}
            hint={H.gridsearch} onClick={() => setGridOpen(true)}
            disabled={busy || training}
            className="mt-[11px]"
          >
            Buscar mejores parámetros
          </Button>

          <Button
            full
            icon={<LayersIcon />}
            hint={H.structure} onClick={() => setArchOpen(true)}
            className="mt-[11px]"
          >
            Ver la estructura
          </Button>

          <Button
            full
            icon={<SparklesIcon />}
            hint={H.metrics} onClick={() => setMetricsOpen(true)}
            disabled={!trained || busy || training || outdated || archMissing}
            className="mt-[11px]"
          >
            Ver métricas
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
        />
      )}

      {archOpen && (
        <ArchitectureModal
          arch={aeArchitecture(arch)}
          latentDim={latentDim}
          onClose={() => setArchOpen(false)}
        />
      )}

      {metricsOpen && (
        <MetricsPanel arch={arch} latentDim={latentDim} onClose={() => setMetricsOpen(false)} />
      )}
    </>
  )
}
