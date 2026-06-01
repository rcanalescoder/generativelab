import type { SVGProps } from 'react'

/*
 * Iconografía del laboratorio. Extraída de /design/mockup.html.
 * Todos son trazos sobre `currentColor`, así heredan el color del contexto.
 * El tamaño lo fija normalmente el CSS del contenedor (.gm-btn svg, .gm-info svg…);
 * para usos sueltos se puede pasar width/height/className por props.
 */
type IconProps = SVGProps<SVGSVGElement>

const base: IconProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
}

export function LogoMark(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" fill="none" {...props}>
      <path
        d="M12 2l8 4.5v9L12 22l-8-6.5v-9L12 2z"
        stroke="#fff"
        strokeWidth={1.6}
        strokeLinejoin="round"
      />
      <path
        d="M12 7v10M7.5 9.5L12 12l4.5-2.5"
        stroke="#fff"
        strokeWidth={1.4}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.85}
      />
    </svg>
  )
}

export function InfoIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 11v5M12 8h.01" strokeLinecap="round" />
    </svg>
  )
}

export function CameraIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M3 7a2 2 0 012-2h2l1.5-2h7L20 5h-1a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V7z"
        strokeLinejoin="round"
      />
      <circle cx="12" cy="12" r="3.5" />
    </svg>
  )
}

export function DownloadIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M12 3v12m0 0l-4-4m4 4l4-4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function DatabaseIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <ellipse cx="12" cy="6" rx="7" ry="3" />
      <path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3" />
    </svg>
  )
}

export function ImageIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3" y="4" width="18" height="16" rx="2" />
      <path d="M3 15l5-4 4 3 3-3 6 5" strokeLinejoin="round" />
      <circle cx="8" cy="9" r="1.4" />
    </svg>
  )
}

export function UploadIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M12 16V5m0 0L8 9m4-4l4 4M5 18v1a2 2 0 002 2h10a2 2 0 002-2v-1"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function RefreshIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M4 8a8 8 0 0114-3m1 3V4m0 4h-4M20 16a8 8 0 01-14 3m-1-3v4m0-4h4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function PlayIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M7 5l11 7-11 7V5z" strokeLinejoin="round" />
    </svg>
  )
}

export function SearchIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="11" cy="11" r="6" />
      <path d="M20 20l-4-4" strokeLinecap="round" />
    </svg>
  )
}

export function InterpolateIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M4 18c5 0 5-12 10-12M14 6h6m0 0l-3-3m3 3l-3 3M4 18h2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function ClustersIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <circle cx="7" cy="8" r="3" />
      <circle cx="17" cy="8" r="3" />
      <circle cx="12" cy="16" r="3" />
    </svg>
  )
}

export function ChevronDownIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M6 9l6 6 6-6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function SparkIcon(props: IconProps) {
  return (
    <svg viewBox="0 0 20 14" fill="none" stroke="currentColor" strokeWidth={1.8} {...props}>
      <path d="M1 9l3-5 3 7 3-9 3 5 3-3 3 4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function SparklesIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M12 3l1.8 4.7L18.5 9.5 13.8 11.3 12 16l-1.8-4.7L5.5 9.5l4.7-1.8L12 3z"
        strokeLinejoin="round"
      />
      <path d="M19 14l.8 2.2 2.2.8-2.2.8L19 20l-.8-2.2-2.2-.8 2.2-.8L19 14z" strokeLinejoin="round" />
    </svg>
  )
}

export function CloseIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M6 6l12 12M18 6L6 18" strokeLinecap="round" />
    </svg>
  )
}

export function SlidersIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h8M16 18h4" strokeLinecap="round" />
      <circle cx="16" cy="6" r="2" />
      <circle cx="8" cy="12" r="2" />
      <circle cx="14" cy="18" r="2" />
    </svg>
  )
}

export function LayersIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M12 3l9 5-9 5-9-5 9-5z" strokeLinejoin="round" />
      <path d="M3 13l9 5 9-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}

export function EyeIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7z" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  )
}

export function WrenchIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path
        d="M14.7 6.3a4 4 0 00-5.2 5.2l-6 6a1.5 1.5 0 002.1 2.1l6-6a4 4 0 005.2-5.2l-2.5 2.5-2.1-2.1 2.5-2.5z"
        strokeLinejoin="round"
      />
    </svg>
  )
}

export function MapIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M9 4L3 6v14l6-2 6 2 6-2V4l-6 2-6-2z" strokeLinejoin="round" />
      <path d="M9 4v14M15 6v14" strokeLinecap="round" />
    </svg>
  )
}

export function GridIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <rect x="3.5" y="3.5" width="7" height="7" rx="1.5" />
      <rect x="13.5" y="3.5" width="7" height="7" rx="1.5" />
      <rect x="3.5" y="13.5" width="7" height="7" rx="1.5" />
      <rect x="13.5" y="13.5" width="7" height="7" rx="1.5" />
    </svg>
  )
}

export function BookIcon(props: IconProps) {
  return (
    <svg {...base} {...props}>
      <path d="M4 5a2 2 0 012-2h13v15H6a2 2 0 00-2 2V5z" strokeLinejoin="round" />
      <path d="M19 18H6a2 2 0 00-2 2" strokeLinecap="round" />
    </svg>
  )
}
