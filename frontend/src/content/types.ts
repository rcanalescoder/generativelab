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

export interface ModelContent {
  /** Guía general de la página (botón "i" del header). */
  general: Topic
  /** Temas por sección, indexados por clave. */
  topics: Record<string, Topic>
}
