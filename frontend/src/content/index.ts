import type { ModelContent } from './types'
import autoencoderJson from './autoencoder.json'

// El JSON se valida por convención (lo escribimos a mano siguiendo el esquema).
export const autoencoderContent = autoencoderJson as unknown as ModelContent

/** Contenidos "i" por lengüeta. Se irá rellenando al construir cada modelo. */
export const CONTENT: Partial<Record<string, ModelContent>> = {
  autoencoder: autoencoderContent,
}
