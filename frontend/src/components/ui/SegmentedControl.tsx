import { cn } from '../../lib/cn'

interface Option<T extends string> {
  label: string
  value: T
}

interface SegmentedControlProps<T extends string> {
  options: Option<T>[]
  value: T
  onChange: (value: T) => void
  className?: string
}

/** Control segmentado tipo MSE/L1. Genérico sobre el conjunto de valores. */
export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  className,
}: SegmentedControlProps<T>) {
  return (
    <span className={cn('gm-seg', className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          className={cn(o.value === value && 'is-on')}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </span>
  )
}
