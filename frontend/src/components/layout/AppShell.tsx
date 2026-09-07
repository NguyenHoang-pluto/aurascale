import type { ReactNode } from 'react'
import { TopNav } from './TopNav'

/**
 * Application frame: a sticky top bar over a single work area.
 *
 * `h-dvh` with `overflow-hidden` rather than `min-h-dvh`: the workspace manages
 * its own scrolling regions so the image canvas can own the viewport height
 * without the page scrolling behind it.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-dvh flex-col overflow-hidden bg-background">
      <a
        href="#main"
        className={
          'sr-only focus:not-sr-only focus:absolute focus:top-3 focus:left-3 focus:z-50 ' +
          'focus:rounded-md focus:border focus:border-border focus:bg-surface-raised ' +
          'focus:px-3 focus:py-2 focus:text-sm focus:font-medium'
        }
      >
        Skip to content
      </a>
      <TopNav />
      <main id="main" className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {children}
      </main>
    </div>
  )
}
