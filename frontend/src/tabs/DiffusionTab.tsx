import { ComingSoon } from '../components/lab/ComingSoon'

export function DiffusionTab() {
  return (
    <ComingSoon
      phase="Fase 5"
      title="Diffusion · DDPM"
      description="Proceso forward (añadir ruido) y reverse (denoising). El schedule, la predicción de ruido ε y el muestreo iterativo con DDPM/DDIM."
    />
  )
}
