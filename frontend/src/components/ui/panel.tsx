import type { ComponentPropsWithoutRef, ElementType } from 'react'
import { cn } from '@/lib/cn'

/**
 * A bordered surface. The workspace uses these for the settings rail and
 * information panels.
 *
 * Deliberately restrained: a 1px border and a flat surface colour rather than
 * a large radius and a drop shadow. The image is the subject of this product,
 * so panels recede.
 */
export function Panel({
  className,
  as,
  ...props
}: ComponentPropsWithoutRef<'div'> & { as?: ElementType }) {
  const Component = as ?? 'div'
  return (
    <Component
      className={cn('rounded-lg border border-border bg-surface', className)}
      {...props}
    />
  )
}

export function PanelHeader({ className, ...props }: ComponentPropsWithoutRef<'div'>) {
  return (
    <div
      className={cn('flex items-center justify-between gap-2 px-4 py-3', className)}
      {...props}
    />
  )
}

export function PanelTitle({ className, ...props }: ComponentPropsWithoutRef<'h2'>) {
  return (
    <h2
      className={cn(
        'text-xs font-semibold uppercase tracking-wider text-muted-foreground',
        className,
      )}
      {...props}
    />
  )
}

export function PanelContent({ className, ...props }: ComponentPropsWithoutRef<'div'>) {
  return <div className={cn('px-4 pb-4', className)} {...props} />
}

export function PanelFooter({ className, ...props }: ComponentPropsWithoutRef<'div'>) {
  return (
    <div
      className={cn('flex items-center gap-2 border-t border-border px-4 py-3', className)}
      {...props}
    />
  )
}
