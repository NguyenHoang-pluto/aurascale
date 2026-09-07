import type { LucideIcon } from 'lucide-react'
import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

/**
 * Centred empty/idle state for a region that has no content yet.
 *
 * Kept quiet on purpose: no illustration, no gradient. The workspace should
 * read as a tool waiting for input, not a marketing page.
 */
export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: {
  icon: LucideIcon
  title: string
  description?: string
  action?: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center px-6 py-12 text-center',
        className,
      )}
    >
      <div className="grid size-11 place-items-center rounded-lg border border-border bg-surface">
        <Icon aria-hidden="true" className="size-5 text-muted-foreground" />
      </div>
      <h2 className="mt-4 text-sm font-semibold tracking-tight text-foreground">{title}</h2>
      {description !== undefined && (
        <p className="mt-1.5 max-w-sm text-sm text-muted-foreground">{description}</p>
      )}
      {action !== undefined && <div className="mt-4">{action}</div>}
    </div>
  )
}
