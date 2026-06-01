import { useEffect, useRef, useState } from 'react'
import { Header } from './layout/Header'
import { TABS, type TabId } from './tabs/types'
import { AutoencoderTab } from './tabs/AutoencoderTab'
import { VAETab } from './tabs/VAETab'
import { GANTab } from './tabs/GANTab'
import { DiffusionTab } from './tabs/DiffusionTab'
import { ComparatorTab } from './tabs/ComparatorTab'
import { InfoProvider } from './components/info/InfoProvider'
import { CONTENT } from './content'
import { cn } from './lib/cn'
import { exportNodeToPng } from './lib/exportPng'
import { CloseIcon, DownloadIcon } from './lib/icons'

/** La lengüeta activa se refleja en el hash de la URL (#vae, #gan…) para enlazar/recargar. */
function tabFromHash(): TabId {
  const h = window.location.hash.replace('#', '')
  return TABS.some((t) => t.id === h) ? (h as TabId) : 'autoencoder'
}

function renderTab(id: TabId) {
  switch (id) {
    case 'autoencoder':
      return <AutoencoderTab />
    case 'vae':
      return <VAETab />
    case 'gan':
      return <GANTab />
    case 'diffusion':
      return <DiffusionTab />
    case 'comparador':
      return <ComparatorTab />
  }
}

function App() {
  const [active, setActive] = useState<TabId>(tabFromHash)
  const [capture, setCapture] = useState(false)
  const mainRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const onHash = () => setActive(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const change = (id: TabId) => {
    window.location.hash = id
    setActive(id)
  }

  const exportPng = () => {
    if (mainRef.current) exportNodeToPng(mainRef.current, `gmvl-${active}`)
  }

  return (
    <InfoProvider>
      <div className={cn('gm-wrap', capture && 'gm-capture')}>
        <Header
          active={active}
          onChange={change}
          general={CONTENT[active]?.general ?? null}
          capture={capture}
          onToggleCapture={() => setCapture((c) => !c)}
          onExport={exportPng}
        />
        {/* key fuerza el remount al cambiar de lengüeta → reaparece la animación de entrada */}
        <main ref={mainRef} key={active}>
          {renderTab(active)}
        </main>
      </div>

      {capture && (
        <div className="gm-capture-bar" data-no-export="true">
          <button type="button" className="gm-capture-exit" onClick={exportPng}>
            <DownloadIcon /> Exportar PNG
          </button>
          <button type="button" className="gm-capture-exit" onClick={() => setCapture(false)}>
            <CloseIcon /> Salir de modo captura
          </button>
        </div>
      )}
    </InfoProvider>
  )
}

export default App
