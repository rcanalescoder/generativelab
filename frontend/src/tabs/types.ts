/** Identificadores de las cinco lengüetas del laboratorio. */
export type TabId = 'autoencoder' | 'vae' | 'gan' | 'diffusion' | 'comparador'

export interface TabDef {
  id: TabId
  label: string
}

/** Orden y etiquetas del header (CLAUDE.md §0 / maqueta). */
export const TABS: TabDef[] = [
  { id: 'autoencoder', label: 'Autoencoder' },
  { id: 'vae', label: 'VAE' },
  { id: 'gan', label: 'GAN' },
  { id: 'diffusion', label: 'Diffusion' },
  { id: 'comparador', label: 'Comparador' },
]
