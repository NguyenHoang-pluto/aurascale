import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

/** Scrollable container for the non-workspace screens. */
export function PageContainer({
  title,
  description,
  actions,
  children,
  className,
}: {
  title: string
  description?: string
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className={cn('mx-auto w-full max-w-5xl px-4 py-8 sm:px-6', className)}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
            {description !== undefined && (
              <p className="mt-1 text-sm text-muted-foreground">{description}</p>
            )}
          </div>
          {actions}
        </div>
        <div className="mt-6">{children}</div>
      </div>
    </div>
  )
}
