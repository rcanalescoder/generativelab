import { useEffect, useState } from 'react'
import { Button, Card, CardHeader, Skeleton } from '../components/ui'
import { RefreshIcon } from '../lib/icons'
import { useInfo } from '../components/info/InfoProvider'
import { comparadorContent } from '../content'
import { reconstruct } from '../lib/api'
import { generateVAE } from '../lib/api-vae'
import { generateGAN } from '../lib/api-gan'
import { generateDiffusion } from '../lib/api-diffusion'

const C = comparadorContent
const AE_IDS = [12, 800, 4096, 20000, 30000, 41000]
const N = 6

const TABLE: Array<{ prop: string; ae: string; vae: string; gan: string; diffusion: string }> = [
  { prop: '¿Tiene encoder?', ae: 'Sí', vae: 'Sí', gan: 'No', diffusion: 'No' },
  { prop: '¿Genera caras nuevas?', ae: 'No (reconstruye)', vae: 'Sí (del prior)', gan: 'Sí', diffusion: 'Sí' },
  { prop: 'Espacio latente', ae: 'Determinista', vae: 'Probabilístico N(0,I)', gan: 'Ruido z', diffusion: 'Ruido + tiempo t' },
  { prop: 'Entrenamiento', ae: 'Estable', vae: 'Estable', gan: 'Inestable (adversarial)', diffusion: 'Estable' },
  { prop: 'Muestreo', ae: '1 paso', vae: '1 paso', gan: '1 paso', diffusion: 'Iterativo (lento)' },
  { prop: 'Nitidez típica', ae: 'Borrosa', vae: 'Borrosa/suave', gan: 'Nítida (con artefactos)', diffusion: 'Buena' },
]

const MODELS = [
  { key: 'ae', name: 'Autoencoder', note: 'Reconstrucción · no genera caras nuevas', color: '#2563EB' },
  { key: 'vae', name: 'VAE', note: 'Generadas · muestreo del prior', color: '#7C3AED' },
  { key: 'gan', name: 'GAN', note: 'Generadas · desde ruido z', color: '#DB2777' },
  { key: 'diffusion', name: 'Diffusion', note: 'Generadas · denoising', color: '#0891B2' },
] as const

type Imgs = Record<string, string[] | null>

export function ComparatorTab() {
  const { open } = useInfo()
  const [seed, setSeed] = useState(42)
  const [imgs, setImgs] = useState<Imgs>({ ae: null, vae: null, gan: null, diffusion: null })
  const [loading, setLoading] = useState(true)

  const loadAll = async (s: number) => {
    setLoading(true)
    const [ae, vae, gan, diffusion] = await Promise.all([
      reconstruct(AE_IDS, 0).then((items) => items.map((i) => i.reconstruction)).catch(() => null),
      generateVAE(N, s).catch(() => null),
      generateGAN(N, s).catch(() => null),
      generateDiffusion(N, s, 40).catch(() => null),
    ])
    setImgs({ ae, vae, gan, diffusion })
    setLoading(false)
  }

  useEffect(() => {
    loadAll(42)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const cell = (v: string, strong = false) => (
    <td className={`px-3 py-2.5 text-[13px] ${strong ? 'font-medium text-[color:var(--ink)]' : 'text-[color:var(--slate)]'}`}>
      {v}
    </td>
  )

  return (
    <>
      {/* Tabla comparativa */}
      <Card rise={1} className="mb-[22px]">
        <CardHeader title="Tabla comparativa" onInfo={() => open(C.topics.tabla)} infoLabel="Cómo leer la tabla" />
        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-[color:var(--line)]">
                <th className="px-3 py-2 text-left text-[11px] font-semibold uppercase tracking-wide text-[color:var(--muted)]">
                  Propiedad
                </th>
                {MODELS.map((m) => (
                  <th
                    key={m.key}
                    className="px-3 py-2 text-left text-[12.5px] font-semibold"
                    style={{ color: m.color }}
                  >
                    {m.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TABLE.map((row) => (
                <tr key={row.prop} className="border-b border-[#f1f2f5]">
                  {cell(row.prop, true)}
                  {cell(row.ae)}
                  {cell(row.vae)}
                  {cell(row.gan)}
                  {cell(row.diffusion)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Galería comparada */}
      <Card rise={2}>
        <CardHeader
          title="Galería comparada"
          onInfo={() => open(C.topics.galeria)}
          infoLabel="Qué estás viendo"
          actions={
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-[12.5px] text-[color:var(--slate)]">
                seed
                <input
                  type="number"
                  className="gm-val"
                  style={{ width: 64 }}
                  value={seed}
                  onChange={(e) => setSeed(Number(e.target.value))}
                />
              </label>
              <Button icon={<RefreshIcon />} onClick={() => loadAll(seed)} disabled={loading}>
                Generar en todos
              </Button>
            </div>
          }
        />

        <div className="flex flex-col gap-5">
          {MODELS.map((m) => {
            const row = imgs[m.key]
            return (
              <div key={m.key}>
                <div className="mb-2 flex items-baseline gap-2.5">
                  <span className="text-[14.5px] font-semibold" style={{ color: m.color }}>
                    {m.name}
                  </span>
                  <span className="text-[12px] text-[color:var(--muted)]">{m.note}</span>
                </div>
                <div className="grid grid-cols-6 gap-2.5">
                  {loading || row === null
                    ? Array.from({ length: N }).map((_, i) =>
                        loading ? (
                          <Skeleton key={i} className="aspect-square w-full !rounded-[12px]" />
                        ) : (
                          <div
                            key={i}
                            className="flex aspect-square w-full items-center justify-center rounded-[12px] border border-dashed border-[color:var(--line)] text-[11px] text-[color:var(--muted)]"
                          >
                            {i === 2 ? 'sin entrenar' : ''}
                          </div>
                        ),
                      )
                    : row.map((src, i) => (
                        <div
                          key={i}
                          className="aspect-square w-full overflow-hidden rounded-[12px] border border-[rgba(17,24,39,0.06)]"
                        >
                          <img src={src} alt={`${m.name} ${i}`} className="h-full w-full" />
                        </div>
                      ))}
                </div>
              </div>
            )
          })}
        </div>
      </Card>
    </>
  )
}
