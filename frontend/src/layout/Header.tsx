import { Button, Tabs } from '../components/ui'
import { BookIcon, CameraIcon, DownloadIcon, LogoMark } from '../lib/icons'
import { cn } from '../lib/cn'
import { useHealth } from '../lib/useHealth'
import { useInfo } from '../components/info/InfoProvider'
import { TABS, type TabId } from '../tabs/types'
import type { Topic } from '../content/types'

/** Pill de estado del backend: MPS (verde, pulsa), CPU (ámbar) o sin conexión. */
function MpsPill() {
  const { health, error, loading } = useHealth()

  if (loading) {
    return (
      <span className="gm-pill-mps is-off" data-chrome="true">
        <span className="gm-dot" />
        Conectando…
      </span>
    )
  }
  if (error || !health) {
    return (
      <span className="gm-pill-mps is-off" data-chrome="true" title={error ?? 'Sin respuesta del backend'}>
        <span className="gm-dot" />
        Backend sin conexión
      </span>
    )
  }
  const isMps = health.device === 'mps'
  return (
    <span
      className={cn('gm-pill-mps', !isMps && 'is-cpu')}
      data-chrome="true"
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
  /** Guía general de la lengüeta activa (botón "i" del header). */
  general: Topic | null
  capture: boolean
  onToggleCapture: () => void
  onExport: () => void
}

export function Header({ active, onChange, general, capture, onToggleCapture, onExport }: HeaderProps) {
  const { open } = useInfo()

  return (
    <header className="mb-[26px] flex items-center justify-between gap-4">
      <div className="flex items-center">
        <div className="flex items-center gap-[13px]">
          <div className="gm-brand-logo">
            <LogoMark className="h-[21px] w-[21px]" />
          </div>
          <h1 className="text-[19px] font-semibold tracking-[-0.3px]">Generative Models Visual Lab</h1>
        </div>
        <Tabs items={TABS} active={active} onChange={onChange} className="ml-[30px]" />
      </div>

      <div className="flex items-center gap-[11px]">
        <MpsPill />
        {general && (
          <span data-chrome="true">
            <Button icon={<BookIcon />} onClick={() => open(general)} title="Guía completa de esta página">
              Guía
            </Button>
          </span>
        )}
        <span data-chrome="true">
          <Button
            icon={<CameraIcon />}
            onClick={onToggleCapture}
            title="Oculta el cromo para hacer capturas limpias"
          >
            Modo captura
          </Button>
        </span>
        <Button icon={<DownloadIcon />} onClick={onExport} title={capture ? 'Exportar PNG' : 'Exporta la vista actual a PNG'}>
          Exportar PNG
        </Button>
      </div>
    </header>
  )
}
