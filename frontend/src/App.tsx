import { useEffect, useState } from 'react'
import { Header } from './layout/Header'
import { TABS, type TabId } from './tabs/types'
import { AutoencoderTab } from './tabs/AutoencoderTab'
import { VAETab } from './tabs/VAETab'
import { GANTab } from './tabs/GANTab'
import { DiffusionTab } from './tabs/DiffusionTab'
import { ComparatorTab } from './tabs/ComparatorTab'

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

  useEffect(() => {
    const onHash = () => setActive(tabFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  const change = (id: TabId) => {
    window.location.hash = id
    setActive(id)
  }

  return (
    <div className="gm-wrap">
      <Header active={active} onChange={change} />
      {/* key fuerza el remount al cambiar de lengüeta → reaparece la animación de entrada */}
      <main key={active}>{renderTab(active)}</main>
    </div>
  )
}

export default App
