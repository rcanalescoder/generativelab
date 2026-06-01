import type { ModelContent } from './types'
import autoencoderJson from './autoencoder.json'
import vaeJson from './vae.json'
import ganJson from './gan.json'
import diffusionJson from './diffusion.json'
import comparadorJson from './comparador.json'

// El JSON se valida por convención (lo escribimos a mano siguiendo el esquema).
export const autoencoderContent = autoencoderJson as unknown as ModelContent
export const vaeContent = vaeJson as unknown as ModelContent
export const ganContent = ganJson as unknown as ModelContent
export const diffusionContent = diffusionJson as unknown as ModelContent
export const comparadorContent = comparadorJson as unknown as ModelContent

/** Contenidos "i" por lengüeta. */
export const CONTENT: Partial<Record<string, ModelContent>> = {
  autoencoder: autoencoderContent,
  vae: vaeContent,
  gan: ganContent,
  diffusion: diffusionContent,
  comparador: comparadorContent,
}
