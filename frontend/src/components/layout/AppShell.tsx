import type { ReactNode } from 'react'
import { TopNav } from './TopNav'

/**
 * Application frame: fixed top navigation over a single scrolling work area.
 * The workspace owns its own layout so the image viewer can claim full height.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-dvh flex-col bg-background">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-surface focus:px-3 focus:py-2 focus:text-sm"
      >
        Skip to content
      </a>
      <TopNav />
      <main id="main" className="flex flex-1 flex-col">
        {children}
      </main>
    </div>
  )
}
