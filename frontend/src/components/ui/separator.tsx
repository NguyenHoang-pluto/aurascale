import type { ComponentPropsWithoutRef } from 'react'
import { cn } from '@/lib/cn'

/**
 * Visual divider. Decorative by default, so screen readers skip it; pass
 * `decorative={false}` when the separation carries meaning.
 */
export function Separator({
  className,
  orientation = 'horizontal',
  decorative = true,
  ...props
}: ComponentPropsWithoutRef<'div'> & {
  orientation?: 'horizontal' | 'vertical'
  decorative?: boolean
}) {
  return (
    <div
      className={cn(
        'shrink-0 bg-border',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className,
      )}
      {...(decorative
        ? { 'aria-hidden': true }
        : { role: 'separator', 'aria-orientation': orientation })}
      {...props}
    />
  )
}
