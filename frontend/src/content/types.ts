// Esquema de los contenidos "i" (la teoría vive en JSON, no en código — CLAUDE.md).
// Un InfoPanel renderiza un Topic: título + apartados, cada uno con su color y bloques.

export type Accent = 'blue' | 'violet' | 'pink' | 'green' | 'amber' | 'cyan'

export type Block =
  | { type: 'p'; text: string }
  | { type: 'list'; items: string[] }
  | { type: 'defs'; items: { term: string; desc: string }[] }
  | { type: 'steps'; items: string[] }
  | { type: 'note'; tone?: 'info' | 'warn' | 'success'; text: string }
  | { type: 'formula'; text: string }

export interface Section {
  heading: string
  accent: Accent
  /** Clave de icono (ver mapa en InfoPanel). Opcional. */
  icon?: string
  blocks: Block[]
}

export interface Topic {
  title: string
  subtitle?: string
  /** Etiqueta corta para el encabezado (p.ej. "Guía", "Mapa latente"). */
  badge?: string
  accent?: Accent
  sections: Section[]
}

/** Categoría de un control: marca si tocar el control obliga a reentrenar, surte efecto
 *  inmediato, o es una acción que lanza algo. El tooltip la muestra como chip de color. */
export type HintKind = 'retrain' | 'live' | 'action'

/** Ayuda contextual de UN control o botón (el tooltip ⓘ). Vive en JSON, como toda la teoría. */
export interface ControlHint {
  /** Qué hace y qué efecto esperar, en 1-2 frases. */
  text: string
  kind?: HintKind
}

export interface ModelContent {
  /** Guía general de la página (botón "i" del header). */
  general: Topic
  /** Temas por sección, indexados por clave. */
  topics: Record<string, Topic>
  /** Ayudas contextuales por control (tooltips ⓘ), indexadas por clave de control. */
  hints?: Record<string, ControlHint>
}
