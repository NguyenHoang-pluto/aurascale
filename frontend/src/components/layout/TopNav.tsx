import { NavLink } from 'react-router-dom'
import { Moon, Sun } from 'lucide-react'
import { cn } from '@/lib/cn'
import { useThemeStore } from '@/stores/useThemeStore'

const NAV_ITEMS = [
  { to: '/', label: 'Enhance', end: true },
  { to: '/history', label: 'History', end: false },
  { to: '/settings', label: 'Settings', end: false },
] as const

export function TopNav() {
  const theme = useThemeStore((s) => s.theme)
  const toggleTheme = useThemeStore((s) => s.toggleTheme)

  return (
    <header className="border-b border-border bg-surface/80 backdrop-blur-sm">
      <div className="flex h-14 items-center gap-6 px-4 sm:px-6">
        <a href="/" className="flex items-center gap-2 rounded-md" aria-label="PixelForge AI home">
          <span
            aria-hidden="true"
            className="grid size-7 place-items-center rounded-md bg-accent text-accent-foreground font-mono text-xs font-bold"
          >
            PF
          </span>
          <span className="text-sm font-semibold tracking-tight">PixelForge AI</span>
        </a>

        <nav aria-label="Main" className="flex items-center gap-1">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                cn(
                  'rounded-md px-3 py-1.5 text-sm transition-colors',
                  'hover:bg-muted hover:text-foreground',
                  isActive ? 'bg-muted text-foreground' : 'text-muted-foreground',
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          {/* GPU / backend status indicator is added in Phase 5, once /api/system exists. */}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
            className={cn(
              'grid size-8 place-items-center rounded-md text-muted-foreground',
              'transition-colors hover:bg-muted hover:text-foreground',
            )}
          >
            {theme === 'dark' ? (
              <Sun className="size-4" aria-hidden="true" />
            ) : (
              <Moon className="size-4" aria-hidden="true" />
            )}
          </button>
        </div>
      </div>
    </header>
  )
}
