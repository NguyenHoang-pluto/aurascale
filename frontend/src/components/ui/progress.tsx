import { cn } from '@/lib/cn'

/**
 * Determinate progress bar.
 *
 * `value` must be a real measurement — this component is used for job progress,
 * which is derived from completed tiles, never from a timer.
 */
export function Progress({
  value,
  label,
  className,
}: {
  value: number
  /** Accessible name. Required: a bare progressbar tells a screen reader nothing. */
  label: string
  className?: string
}) {
  const clamped = Math.min(100, Math.max(0, Number.isFinite(value) ? value : 0))

  return (
    <div
      role="progressbar"
      aria-valuenow={Math.round(clamped)}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      className={cn('h-1.5 w-full overflow-hidden rounded-full bg-muted', className)}
    >
      <div
        className="h-full rounded-full bg-accent transition-[width] duration-300 ease-out"
        style={{ width: `${clamped}%` }}
      />
    </div>
  )
}
