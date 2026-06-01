import { cn } from '../../lib/cn'

interface TabItem<T extends string> {
  id: T
  label: string
}

interface TabsProps<T extends string> {
  items: TabItem<T>[]
  active: T
  onChange: (id: T) => void
  className?: string
}

/** Lengüetas de navegación del header (subrayado azul en la activa). */
export function Tabs<T extends string>({ items, active, onChange, className }: TabsProps<T>) {
  return (
    <nav className={cn('flex items-center gap-1.5', className)}>
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          className={cn('gm-nav-link', it.id === active && 'is-active')}
          onClick={() => onChange(it.id)}
        >
          {it.label}
        </button>
      ))}
    </nav>
  )
}
