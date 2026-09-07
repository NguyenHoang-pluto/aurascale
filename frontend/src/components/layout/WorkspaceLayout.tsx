import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

/**
 * The enhancement workspace: a dominant canvas with a settings rail beside it.
 *
 * The canvas is the subject of this product (§ 31), so it takes all remaining
 * space and the rail is a fixed, modest width. Below `lg` the rail moves under
 * the canvas rather than shrinking, because a 240px settings column is not
 * usable and squeezing the canvas defeats the point of the screen.
 *
 * `min-h-0` / `min-w-0` on the flex children is deliberate: without it a flex
 * item refuses to shrink below its content size and a large image forces
 * horizontal page overflow, which § 18 forbids.
 */
export function WorkspaceLayout({
  canvas,
  rail,
  toolbar,
  className,
}: {
  canvas: ReactNode
  rail: ReactNode
  /** Optional strip directly above the canvas — zoom and comparison controls. */
  toolbar?: ReactNode
  className?: string
}) {
  return (
    <div className={cn('flex min-h-0 flex-1 flex-col lg:flex-row', className)}>
      <section
        aria-label="Image workspace"
        className="flex min-h-0 min-w-0 flex-1 flex-col"
      >
        {toolbar !== undefined && (
          <div className="flex shrink-0 items-center gap-2 border-b border-border px-4 py-2">
            {toolbar}
          </div>
        )}
        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-canvas">
          {canvas}
        </div>
      </section>

      <aside
        aria-label="Enhancement settings"
        className={cn(
          'shrink-0 overflow-y-auto border-border bg-background',
          'border-t lg:w-[320px] lg:border-t-0 lg:border-l xl:w-[360px]',
        )}
      >
        <div className="flex flex-col gap-4 p-4">{rail}</div>
      </aside>
    </div>
  )
}
