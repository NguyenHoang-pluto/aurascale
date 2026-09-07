import { Menu, Moon, Settings, Sun, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tooltip } from '@/components/ui/tooltip'
import { BackendStatusIndicator } from '@/features/system/BackendStatusIndicator'
import { MOBILE_QUERY, useMediaQuery } from '@/hooks/useMediaQuery'
import { cn } from '@/lib/cn'
import { useThemeStore } from '@/stores/useThemeStore'

const NAV_ITEMS = [
  { to: '/', label: 'Enhance', end: true },
  { to: '/history', label: 'History', end: false },
  { to: '/settings', label: 'Settings', end: false },
] as const

function navLinkClass({ isActive }: { isActive: boolean }): string {
  return cn(
    'rounded-md px-3 py-1.5 text-sm transition-colors',
    'hover:bg-muted hover:text-foreground',
    isActive ? 'bg-muted font-medium text-foreground' : 'text-muted-foreground',
  )
}

function ThemeToggle() {
  const resolved = useThemeStore((s) => s.resolved)
  const toggleTheme = useThemeStore((s) => s.toggleTheme)
  const target = resolved === 'dark' ? 'light' : 'dark'

  return (
    <Tooltip content={`Switch to ${target} theme`}>
      <Button
        variant="ghost"
        size="icon"
        onClick={toggleTheme}
        aria-label={`Switch to ${target} theme`}
      >
        {resolved === 'dark' ? (
          <Sun aria-hidden="true" />
        ) : (
          <Moon aria-hidden="true" />
        )}
      </Button>
    </Tooltip>
  )
}

export function TopNav() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  // Branching in JS rather than hiding with CSS: a display:none nav is
  // removed from the accessibility tree by the browser, but keeping both in
  // the DOM makes the component's contract depend on stylesheets loading.
  const isCompact = useMediaQuery(MOBILE_QUERY)

  return (
    <header className="sticky top-0 z-40 border-b border-border bg-surface/85 backdrop-blur-sm">
      <div className="flex h-14 items-center gap-3 px-4 sm:gap-6 sm:px-6">
        <NavLink
          to="/"
          className="flex shrink-0 items-center gap-2 rounded-md"
          aria-label="PixelForge AI, go to workspace"
        >
          <span
            aria-hidden="true"
            className="grid size-7 place-items-center rounded-md bg-accent font-mono text-xs font-bold text-accent-foreground"
          >
            PF
          </span>
          <span className="text-sm font-semibold tracking-tight">PixelForge AI</span>
        </NavLink>

        {!isCompact && (
          <nav aria-label="Main" className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink key={item.to} to={item.to} end={item.end} className={navLinkClass}>
                {item.label}
              </NavLink>
            ))}
          </nav>
        )}

        <div className="ml-auto flex items-center gap-2">
          <BackendStatusIndicator />
          <Separator orientation="vertical" className="hidden h-5 sm:block" />
          <ThemeToggle />

          {/*
            The settings shortcut is rendered only where the main nav is not
            visible. Showing both would put two links with the identical
            accessible name "Settings" in the same bar, pointing at the same
            route -- redundant for everyone and actively confusing when
            navigating by links with a screen reader.
          */}
          {isCompact && (
            <Tooltip content="Settings">
              <Button variant="ghost" size="icon" asChild>
                <NavLink to="/settings" aria-label="Settings">
                  <Settings aria-hidden="true" />
                </NavLink>
              </Button>
            </Tooltip>
          )}

          {isCompact && (
            <Button
              variant="ghost"
              size="icon"
              aria-label={isMenuOpen ? 'Close navigation menu' : 'Open navigation menu'}
              aria-expanded={isMenuOpen}
              aria-controls="mobile-nav"
              onClick={() => { setIsMenuOpen((open) => !open) }}
            >
              {isMenuOpen ? <X aria-hidden="true" /> : <Menu aria-hidden="true" />}
            </Button>
          )}
        </div>
      </div>

      {isCompact && isMenuOpen && (
        <nav
          id="mobile-nav"
          aria-label="Main"
          className="flex flex-col gap-1 border-t border-border px-4 py-2"
        >
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={navLinkClass}
              // Closed here rather than from a route-change effect: this is
              // the event that causes the navigation, and it also fires when
              // the target route is the current one.
              onClick={() => { setIsMenuOpen(false) }}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      )}
    </header>
  )
}
