import { cn } from '@/lib/cn'

export interface SegmentedOption<T extends string> {
  value: T
  label: string
  /** Shown in a tooltip-free inline hint; keep it to a few words. */
  hint?: string
  /**
   * Native tooltip, for saying why an option is disabled.
   *
   * A control that is greyed out with no explanation reads as a bug rather
   * than a limit, and this is the same affordance the target-size selector
   * already uses for the same purpose.
   */
  title?: string
  disabled?: boolean
}

/**
 * A compact exclusive choice — upscale factor, comparison mode, output format.
 *
 * Built on native radio inputs rather than buttons with `aria-checked`:
 * a radio group gives arrow-key navigation, a single tab stop, and correct
 * announcement ("2 of 3") for free, and it participates in forms.
 */
export function SegmentedControl<T extends string>({
  name,
  value,
  onChange,
  options,
  label,
  className,
  size = 'md',
}: {
  name: string
  value: T
  onChange: (value: T) => void
  options: ReadonlyArray<SegmentedOption<T>>
  /** Accessible group name, e.g. "Upscale factor". */
  label: string
  className?: string
  size?: 'sm' | 'md'
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        'inline-flex w-full items-center gap-1 rounded-md border border-border bg-muted p-1',
        className,
      )}
    >
      {options.map((option) => {
        const isSelected = option.value === value
        const id = `${name}-${option.value}`

        return (
          <label
            key={option.value}
            htmlFor={id}
            {...(option.title !== undefined ? { title: option.title } : {})}
            className={cn(
              'relative flex flex-1 cursor-pointer items-center justify-center rounded',
              'font-medium transition-colors select-none',
              size === 'sm' ? 'h-6 text-xs' : 'h-7 text-sm',
              isSelected
                ? 'bg-surface-raised text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground',
              option.disabled === true && 'cursor-not-allowed opacity-45 hover:text-muted-foreground',
              // The visible ring follows the input's focus, since the input
              // itself is visually hidden.
              'has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2',
              'has-[:focus-visible]:outline-[var(--color-ring)]',
            )}
          >
            <input
              id={id}
              type="radio"
              name={name}
              value={option.value}
              checked={isSelected}
              disabled={option.disabled ?? false}
              onChange={() => {
                onChange(option.value)
              }}
              className="sr-only"
            />
            {option.label}
          </label>
        )
      })}
    </div>
  )
}
