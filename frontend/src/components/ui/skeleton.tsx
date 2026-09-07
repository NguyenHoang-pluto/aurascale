import type { ComponentPropsWithoutRef } from 'react'
import { cn } from '@/lib/cn'

/**
 * Loading placeholder. Marked aria-hidden because the surrounding region
 * should carry aria-busy; announcing a skeleton itself is noise.
 */
export function Skeleton({ className, ...props }: ComponentPropsWithoutRef<'div'>) {
  return (
    <div
      aria-hidden="true"
      className={cn('animate-pulse rounded-md bg-muted', className)}
      {...props}
    />
  )
}
