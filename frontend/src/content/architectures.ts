/**
 * Esquemas de arquitectura de cada red, como DATOS estructurados.
 *
 * Se han extraído fielmente de los modelos reales del backend:
 *   - backend/app/models/ae.py        (ConvAutoencoder, BASE=32, 64×64×3)
 *   - backend/app/models/vae.py       (ConvVAE, mismo stack + cabezas μ/logσ²)
 *   - backend/app/models/gan.py       (Generator + Discriminator DCGAN, BASE=64)
 *   - backend/app/models/diffusion.py (TimeUNet, BASE=64, 32×32×3, T=200)
 *
 * El `ArchitectureModal` consume estos datos y los pinta como un diagrama de
 * columnas con cajas por capa, flechas y colores por sección. Añadir o ajustar
 * una arquitectura es escribir datos aquí, no tocar el componente.
 */

/** Acento de color de una caja/sección (mapea a las clases `gm-acc-*` del CSS). */
export type ArchAccent = 'blue' | 'violet' | 'pink' | 'green' | 'amber' | 'cyan'

/** Una caja del diagrama = una capa (o bloque) de la red. */
export interface ArchLayer {
  /** Nombre corto de la capa, p. ej. "Conv 4×4 /2" o "ConvT 4×4 ×2". */
  nombre: string
  /** Detalle técnico: filtros, stride, norma, activación… p. ej. "32→64 · BatchNorm · LeakyReLU". */
  detalle?: string
  /** Forma del tensor de SALIDA de la capa, p. ej. "32×32×64" o "z (latent_dim)". */
  forma: string
  /** Marca esta caja como destacada (entrada o salida de la red). */
  destacada?: boolean
}

/** Una columna del diagrama (encoder, decoder, generador, una rama de la UNet…). */
export interface ArchColumn {
  /** Encabezado de la columna, p. ej. "ENCODER" o "GENERADOR". */
  titulo: string
  /** Subtítulo opcional bajo el encabezado, p. ej. "imagen → código". */
  subtitulo?: string
  /** Color de la columna. */
  acento: ArchAccent
  /** Capas apiladas de arriba abajo (en orden de flujo). */
  capas: ArchLayer[]
}

/** El bloque latente central (cuello de botella), entre encoder y decoder. */
export interface ArchLatent {
  /** Título del bloque, p. ej. "ESPACIO LATENTE z". */
  titulo: string
  /** Acento del bloque. */
  acento: ArchAccent
  /** Líneas descriptivas (cada una se pinta como un "chip"/fila). */
  lineas: string[]
  /** Si `true`, el componente puede sustituir el marcador {DIM} por el latentDim actual. */
  usaLatentDim?: boolean
}

/** Esquema completo de la arquitectura de un modelo. */
export interface Architecture {
  /** Clave del modelo. */
  modelo: 'ae' | 'vae' | 'gan' | 'diffusion'
  /** Título del diagrama, p. ej. "Autoencoder convolucional". */
  titulo: string
  /** Frase corta de contexto (va en el banner del modal). */
  resumen: string
  /** Resolución/entrada de referencia, p. ej. "Entrada: 64×64×3". */
  entrada: string
  /** Columnas del diagrama (2 o 3). */
  columnas: ArchColumn[]
  /** Bloque latente central, si la arquitectura lo tiene (AE/VAE). */
  latente?: ArchLatent
  /** Notas al pie del diagrama (1–3 líneas pedagógicas). */
  notas?: string[]
}

/* ------------------------------------------------------------------ *
 * AUTOENCODER  (ae.py · BASE=32 · 64×64×3 → z → 64×64×3)
 * encoder: 4 bloques Conv 4×4 stride-2 (cada uno /2): 64→32→16→8→4
 *          canales 3→32→64→128→256, luego flatten + Linear → z
 * decoder: Linear → 4×4×256, 3 bloques ConvT ×2 + ConvT final, Sigmoid
 * ------------------------------------------------------------------ */
const ae: Architecture = {
  modelo: 'ae',
  titulo: 'Autoencoder convolucional',
  resumen:
    'Comprime la imagen a un código z con un cuello de botella y la reconstruye con un decoder simétrico.',
  entrada: 'Entrada: 64×64×3',
  columnas: [
    {
      titulo: 'ENCODER',
      subtitulo: 'imagen → código',
      acento: 'blue',
      capas: [
        { nombre: 'Entrada', detalle: 'imagen RGB', forma: '64×64×3', destacada: true },
        { nombre: 'Conv 4×4 /2', detalle: '3→32 · LeakyReLU(0.2)', forma: '32×32×32' },
        { nombre: 'Conv 4×4 /2', detalle: '32→64 · BatchNorm · LeakyReLU', forma: '16×16×64' },
        { nombre: 'Conv 4×4 /2', detalle: '64→128 · BatchNorm · LeakyReLU', forma: '8×8×128' },
        { nombre: 'Conv 4×4 /2', detalle: '128→256 · BatchNorm · LeakyReLU', forma: '4×4×256' },
        { nombre: 'Flatten + Linear', detalle: '4096 → latent_dim', forma: 'z ({DIM})' },
      ],
    },
    {
      titulo: 'DECODER',
      subtitulo: 'código → imagen',
      acento: 'green',
      capas: [
        { nombre: 'Linear + Reshape', detalle: 'latent_dim → 4096', forma: '4×4×256' },
        { nombre: 'ConvT 4×4 ×2', detalle: '256→128 · BatchNorm · ReLU', forma: '8×8×128' },
        { nombre: 'ConvT 4×4 ×2', detalle: '128→64 · BatchNorm · ReLU', forma: '16×16×64' },
        { nombre: 'ConvT 4×4 ×2', detalle: '64→32 · BatchNorm · ReLU', forma: '32×32×32' },
        { nombre: 'ConvT 4×4 ×2', detalle: '32→3 · Sigmoid → [0,1]', forma: '64×64×3', destacada: true },
      ],
    },
  ],
  latente: {
    titulo: 'ESPACIO LATENTE z',
    acento: 'amber',
    lineas: ['z ∈ ℝ^{DIM}', 'vector determinista', 'el cuello de botella'],
    usaLatentDim: true,
  },
  notas: [
    'Cada Conv stride-2 divide alto y ancho entre 2; cada ConvT los duplica (decoder simétrico).',
    'Pérdida de reconstrucción MSE o L1 entre la imagen original y la reconstruida.',
  ],
}

/* ------------------------------------------------------------------ *
 * VAE  (vae.py · mismo stack que el AE, pero z probabilístico)
 * encoder produce (μ, logσ²); z = μ + σ·ε con ε~N(0,I); decoder idéntico al AE
 * ------------------------------------------------------------------ */
const vae: Architecture = {
  modelo: 'vae',
  titulo: 'Variational Autoencoder',
  resumen:
    'Igual que el AE pero el encoder produce una distribución (μ, logσ²); z se muestrea con el truco de reparametrización.',
  entrada: 'Entrada: 64×64×3',
  columnas: [
    {
      titulo: 'ENCODER',
      subtitulo: 'imagen → (μ, logσ²)',
      acento: 'blue',
      capas: [
        { nombre: 'Entrada', detalle: 'imagen RGB', forma: '64×64×3', destacada: true },
        { nombre: 'Conv 4×4 /2', detalle: '3→32 · LeakyReLU(0.2)', forma: '32×32×32' },
        { nombre: 'Conv 4×4 /2', detalle: '32→64 · BatchNorm · LeakyReLU', forma: '16×16×64' },
        { nombre: 'Conv 4×4 /2', detalle: '64→128 · BatchNorm · LeakyReLU', forma: '8×8×128' },
        { nombre: 'Conv 4×4 /2', detalle: '128→256 · BatchNorm · LeakyReLU', forma: '4×4×256' },
        { nombre: 'Flatten', detalle: '4×4×256', forma: '4096' },
        { nombre: 'Linear μ  ·  Linear logσ²', detalle: '2 cabezas · 4096 → latent_dim', forma: 'μ, logσ² ({DIM})' },
      ],
    },
    {
      titulo: 'DECODER',
      subtitulo: 'código → imagen',
      acento: 'green',
      capas: [
        { nombre: 'Linear + Reshape', detalle: 'latent_dim → 4096', forma: '4×4×256' },
        { nombre: 'ConvT 4×4 ×2', detalle: '256→128 · BatchNorm · ReLU', forma: '8×8×128' },
        { nombre: 'ConvT 4×4 ×2', detalle: '128→64 · BatchNorm · ReLU', forma: '16×16×64' },
        { nombre: 'ConvT 4×4 ×2', detalle: '64→32 · BatchNorm · ReLU', forma: '32×32×32' },
        { nombre: 'ConvT 4×4 ×2', detalle: '32→3 · Sigmoid → [0,1]', forma: '64×64×3', destacada: true },
      ],
    },
  ],
  latente: {
    titulo: 'MUESTREO  z = μ + σ·ε',
    acento: 'violet',
    lineas: ['ε ~ N(0, I)', 'σ = exp(½·logσ²)', 'z ∈ ℝ^{DIM} (reparametrización)'],
    usaLatentDim: true,
  },
  notas: [
    'El encoder no da un punto sino una gaussiana q(z|x); z se muestrea de ella de forma derivable.',
    'Pérdida = reconstrucción (MSE/L1) + β · KL(q(z|x) ‖ N(0,I)).',
  ],
}

/* ------------------------------------------------------------------ *
 * GAN  (gan.py · DCGAN · BASE=64)
 * Generator: z (1×1) → ConvT 4×4 (1→4) + 3×ConvT ×2 + ConvT final → 64×64×3, Tanh
 *            canales z_dim→512→256→128→64→3
 * Discriminator: 64×64×3 → 4×Conv 4×4 /2 + Conv final → 1 logit
 *            canales 3→64→128→256→512→1
 * Sin capas densas. No hay encoder: son dos redes adversarias.
 * ------------------------------------------------------------------ */
const gan: Architecture = {
  modelo: 'gan',
  titulo: 'GAN (DCGAN 64×64)',
  resumen:
    'Dos redes que compiten: el generador sintetiza caras desde ruido y el discriminador intenta distinguir reales de falsas.',
  entrada: 'Generador: z ~ N(0,1)  ·  Discriminador: imagen 64×64×3',
  columnas: [
    {
      titulo: 'GENERADOR',
      subtitulo: 'z → imagen',
      acento: 'pink',
      capas: [
        { nombre: 'Ruido z', detalle: 'z ~ N(0,1) como mapa 1×1', forma: 'z_dim×1×1', destacada: true },
        { nombre: 'ConvT 4×4 /1', detalle: 'z_dim→512 · BatchNorm · ReLU', forma: '4×4×512' },
        { nombre: 'ConvT 4×4 ×2', detalle: '512→256 · BatchNorm · ReLU', forma: '8×8×256' },
        { nombre: 'ConvT 4×4 ×2', detalle: '256→128 · BatchNorm · ReLU', forma: '16×16×128' },
        { nombre: 'ConvT 4×4 ×2', detalle: '128→64 · BatchNorm · ReLU', forma: '32×32×64' },
        { nombre: 'ConvT 4×4 ×2', detalle: '64→3 · Tanh → [-1,1]', forma: '64×64×3', destacada: true },
      ],
    },
    {
      titulo: 'DISCRIMINADOR',
      subtitulo: 'imagen → real / falso',
      acento: 'violet',
      capas: [
        { nombre: 'Entrada', detalle: 'imagen RGB', forma: '64×64×3', destacada: true },
        { nombre: 'Conv 4×4 /2', detalle: '3→64 · LeakyReLU(0.2)', forma: '32×32×64' },
        { nombre: 'Conv 4×4 /2', detalle: '64→128 · BatchNorm · LeakyReLU', forma: '16×16×128' },
        { nombre: 'Conv 4×4 /2', detalle: '128→256 · BatchNorm · LeakyReLU', forma: '8×8×256' },
        { nombre: 'Conv 4×4 /2', detalle: '256→512 · BatchNorm · LeakyReLU', forma: '4×4×512' },
        { nombre: 'Conv 4×4 /1', detalle: '512→1 (sin sigmoid)', forma: '1 logit', destacada: true },
      ],
    },
  ],
  notas: [
    'Arquitectura DCGAN (Radford et al., 2016): sin capas densas, todo convolucional.',
    'El generador (ConvT) aumenta la resolución; el discriminador (Conv stride-2) la reduce: son espejo uno del otro.',
  ],
}

/* ------------------------------------------------------------------ *
 * DIFFUSION  (diffusion.py · TimeUNet · BASE=64 · 32×32×3 · T=200)
 * Embedding temporal sinusoidal → MLP (TIME_DIM=128) inyectado en cada ResBlock.
 * UNet: in_conv → Down1 (32→16) → Down2 (16→8) → mid (8×8) → Up1 (8→16) → Up2 (16→32) → out
 *   c1=64, c2=128, c3=128. Skips: s1 (16×16×128) y s2 (8×8×128).
 * ------------------------------------------------------------------ */
const diffusion: Architecture = {
  modelo: 'diffusion',
  titulo: 'Diffusion (UNet temporal, DDPM)',
  resumen:
    'Una UNet pequeña condicionada en el paso de tiempo t predice el ruido ε de una imagen ruidosa, para luego limpiarla paso a paso.',
  entrada: 'Entrada: x_t 32×32×3  +  paso t',
  columnas: [
    {
      titulo: 'TIEMPO t',
      subtitulo: 'embedding del paso',
      acento: 'amber',
      capas: [
        { nombre: 'Paso t', detalle: 'entero del proceso de ruido', forma: 't', destacada: true },
        { nombre: 'Embedding sinusoidal', detalle: 'senos/cosenos a varias frecuencias', forma: '128' },
        { nombre: 'Linear · SiLU · Linear', detalle: 'MLP del tiempo', forma: '128' },
        { nombre: '→ inyectado en cada ResBlock', detalle: 'FiLM-aditivo (suma por canal)', forma: 'a todos los bloques' },
      ],
    },
    {
      titulo: 'UNet · DOWN',
      subtitulo: 'codificar (bajar resolución)',
      acento: 'blue',
      capas: [
        { nombre: 'Entrada x_t', detalle: 'imagen ruidosa', forma: '32×32×3', destacada: true },
        { nombre: 'Conv 3×3', detalle: '3→64', forma: '32×32×64' },
        { nombre: 'Down 1', detalle: 'ResBlock 64→128 + Conv /2  ⟶ skip s1', forma: '16×16×128' },
        { nombre: 'Down 2', detalle: 'ResBlock 128→128 + Conv /2  ⟶ skip s2', forma: '8×8×128' },
      ],
    },
    {
      titulo: 'UNet · UP',
      subtitulo: 'decodificar (subir + skips)',
      acento: 'green',
      capas: [
        { nombre: 'Bottleneck (mid)', detalle: 'ResBlock 128→128', forma: '8×8×128' },
        { nombre: 'Up 1  ↺ s2', detalle: 'ConvT ×2 + concat s2 → ResBlock →128', forma: '16×16×128' },
        { nombre: 'Up 2  ↺ s1', detalle: 'ConvT ×2 + concat s1 → ResBlock →64', forma: '32×32×64' },
        { nombre: 'GroupNorm · SiLU · Conv 3×3', detalle: '64→3 · ruido ε predicho', forma: '32×32×3', destacada: true },
      ],
    },
  ],
  notas: [
    'Las skip connections (s1, s2) llevan detalle del camino DOWN al UP, concatenándose por canales.',
    'La red predice el ruido ε; el bucle reverse lo resta paso a paso desde ruido puro hasta una cara.',
  ],
}

/** Mapa de arquitecturas por clave de modelo. */
export const ARCHITECTURES = { ae, vae, gan, diffusion } as const

/** Atajo tipado para obtener una arquitectura por modelo. */
export type ArchitectureKey = keyof typeof ARCHITECTURES
