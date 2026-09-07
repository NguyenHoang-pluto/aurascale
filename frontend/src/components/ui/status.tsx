import { AlertTriangle, Check, Loader2, X } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

export type StatusTone = 'idle' | 'pending' | 'success' | 'warning' | 'danger'

const TONE_STYLES: Record<StatusTone, { dot: string; text: string }> = {
  idle: { dot: 'bg-muted-foreground', text: 'text-muted-foreground' },
  pending: { dot: 'bg-accent', text: 'text-muted-foreground' },
  success: { dot: 'bg-success', text: 'text-foreground' },
  warning: { dot: 'bg-warning', text: 'text-foreground' },
  danger: { dot: 'bg-destructive', text: 'text-foreground' },
}

const TONE_ICONS: Record<StatusTone, typeof Check | null> = {
  idle: null,
  pending: Loader2,
  success: Check,
  warning: AlertTriangle,
  danger: X,
}

/**
 * Status indicator that never communicates through colour alone (§ 19).
 *
 * `label` is required and always rendered, and each tone additionally carries a
 * distinct icon. A colour-blind user, a greyscale display and a screen reader
 * all receive the same information.
 */
export function StatusIndicator({
  tone,
  label,
  detail,
  className,
  live = false,
}: {
  tone: StatusTone
  label: string
  detail?: ReactNode
  className?: string
  /** Announce changes to assistive technology. Use for states that change on their own. */
  live?: boolean
}) {
  const styles = TONE_STYLES[tone]
  const Icon = TONE_ICONS[tone]

  return (
    <span
      className={cn('inline-flex items-center gap-2 text-xs', styles.text, className)}
      {...(live ? { role: 'status', 'aria-live': 'polite' } : {})}
    >
      <span className="relative flex size-2 shrink-0 items-center justify-center">
        <span
          aria-hidden="true"
          className={cn(
            'size-2 rounded-full',
            styles.dot,
            tone === 'pending' && 'animate-pulse',
          )}
        />
      </span>
      {Icon !== null && (
        <Icon
          aria-hidden="true"
          className={cn('size-3 shrink-0', tone === 'pending' && 'animate-spin')}
        />
      )}
      <span className="truncate">{label}</span>
      {detail !== undefined && <span className="text-muted-foreground">{detail}</span>}
    </span>
  )
}
