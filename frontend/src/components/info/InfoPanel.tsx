import { useEffect, useRef, type ComponentType, type SVGProps } from 'react'
import type { Block, Section, Topic } from '../../content/types'
import { cn } from '../../lib/cn'
import { exportNodeToPng } from '../../lib/exportPng'
import {
  BookIcon,
  ClustersIcon,
  CloseIcon,
  DatabaseIcon,
  DownloadIcon,
  EyeIcon,
  ImageIcon,
  InfoIcon,
  InterpolateIcon,
  LayersIcon,
  MapIcon,
  SearchIcon,
  SlidersIcon,
  SparklesIcon,
  WrenchIcon,
} from '../../lib/icons'

type IconCmp = ComponentType<SVGProps<SVGSVGElement>>

const ICONS: Record<string, IconCmp> = {
  info: InfoIcon,
  flow: LayersIcon,
  latent: MapIcon,
  results: SparklesIcon,
  sparkles: SparklesIcon,
  sliders: SlidersIcon,
  explore: SearchIcon,
  search: SearchIcon,
  build: WrenchIcon,
  read: EyeIcon,
  dataset: DatabaseIcon,
  image: ImageIcon,
  interpolate: InterpolateIcon,
  clusters: ClustersIcon,
  book: BookIcon,
}

function renderBlock(b: Block, i: number) {
  switch (b.type) {
    case 'p':
      return (
        <p key={i} className="gm-ip-p">
          {b.text}
        </p>
      )
    case 'list':
      return (
        <ul key={i} className="gm-ip-list">
          {b.items.map((t, j) => (
            <li key={j}>{t}</li>
          ))}
        </ul>
      )
    case 'steps':
      return (
        <ol key={i} className="gm-ip-steps">
          {b.items.map((t, j) => (
            <li key={j}>
              <span className="n">{j + 1}</span>
              <span>{t}</span>
            </li>
          ))}
        </ol>
      )
    case 'defs':
      return (
        <dl key={i} className="gm-ip-defs">
          {b.items.map((d, j) => (
            <div key={j}>
              <dt>{d.term}</dt>
              <dd>{d.desc}</dd>
            </div>
          ))}
        </dl>
      )
    case 'note':
      return (
        <div key={i} className={cn('gm-ip-note', `tone-${b.tone ?? 'info'}`)}>
          {b.text}
        </div>
      )
    case 'formula':
      return (
        <div key={i} className="gm-ip-formula">
          {b.text}
        </div>
      )
  }
}

function SectionView({ s }: { s: Section }) {
  const Icon = s.icon ? ICONS[s.icon] : undefined
  return (
    <section className={cn('gm-ip-sec', `gm-acc-${s.accent}`)}>
      <div className="gm-ip-sec-head">
        {Icon && (
          <span className="ic">
            <Icon />
          </span>
        )}
        <h3>{s.heading}</h3>
      </div>
      <div className="gm-ip-sec-body">{s.blocks.map(renderBlock)}</div>
    </section>
  )
}

interface Props {
  topic: Topic
  onClose: () => void
}

export function InfoPanel({ topic, onClose }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const acc = topic.accent ?? 'blue'

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [onClose])

  return (
    <div
      className="gm-ip-overlay"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose()
      }}
    >
      <div className={cn('gm-ip', `gm-acc-${acc}`)} ref={ref} role="dialog" aria-modal="true">
        <div className="gm-ip-banner">
          <div className="gm-ip-banner-text">
            {topic.badge && <span className="gm-ip-badge">{topic.badge}</span>}
            <h2>{topic.title}</h2>
            {topic.subtitle && <p>{topic.subtitle}</p>}
          </div>
          <div className="gm-ip-banner-actions" data-no-export="true">
            <button
              type="button"
              className="gm-ip-x"
              onClick={() => ref.current && exportNodeToPng(ref.current, `info-${topic.title}`)}
              title="Exportar como PNG"
              aria-label="Exportar como PNG"
            >
              <DownloadIcon />
            </button>
            <button type="button" className="gm-ip-x" onClick={onClose} aria-label="Cerrar" title="Cerrar (Esc)">
              <CloseIcon />
            </button>
          </div>
        </div>
        <div className="gm-ip-scroll">
          {topic.sections.map((s, i) => (
            <SectionView key={i} s={s} />
          ))}
        </div>
      </div>
    </div>
  )
}
