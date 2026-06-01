import { Button, Tabs } from '../components/ui'
import { CameraIcon, DownloadIcon, LogoMark } from '../lib/icons'
import { cn } from '../lib/cn'
import { useHealth } from '../lib/useHealth'
import { TABS, type TabId } from '../tabs/types'

/** Pill de estado del backend: MPS (verde, pulsa), CPU (ámbar) o sin conexión. */
function MpsPill() {
  const { health, error, loading } = useHealth()

  if (loading) {
    return (
      <span className="gm-pill-mps is-off">
        <span className="gm-dot" />
        Conectando…
      </span>
    )
  }
  if (error || !health) {
    return (
      <span className="gm-pill-mps is-off" title={error ?? 'Sin respuesta del backend'}>
        <span className="gm-dot" />
        Backend sin conexión
      </span>
    )
  }
  const isMps = health.device === 'mps'
  return (
    <span
      className={cn('gm-pill-mps', !isMps && 'is-cpu')}
      title={`${health.device_name} · PyTorch ${health.torch_version}`}
    >
      <span className="gm-dot" />
      {isMps ? 'MPS activo' : 'CPU'}
    </span>
  )
}

interface HeaderProps {
  active: TabId
  onChange: (id: TabId) => void
}

export function Header({ active, onChange }: HeaderProps) {
  // Los botones de captura llegan en la Fase 2; de momento sin función (GUIA Fase 0).
  const captureSoon = 'Se activa en la Fase 2'

  return (
    <header className="mb-[26px] flex items-center justify-between gap-4">
      <div className="flex items-center">
        <div className="flex items-center gap-[13px]">
          <div className="gm-brand-logo">
            <LogoMark className="h-[21px] w-[21px]" />
          </div>
          <h1 className="text-[19px] font-semibold tracking-[-0.3px]">
            Generative Models Visual Lab
          </h1>
        </div>
        <Tabs items={TABS} active={active} onChange={onChange} className="ml-[30px]" />
      </div>

      <div className="flex items-center gap-[11px]">
        <MpsPill />
        <Button icon={<CameraIcon />} title={captureSoon}>
          Modo captura
        </Button>
        <Button icon={<DownloadIcon />} title={captureSoon}>
          Exportar PNG
        </Button>
      </div>
    </header>
  )
}
