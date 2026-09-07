import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/** What the user chose. `system` follows the OS preference. */
export type ThemePreference = 'dark' | 'light' | 'system'
/** What is actually rendered. `system` is resolved to one of these. */
export type ResolvedTheme = 'dark' | 'light'

const MEDIA_QUERY = '(prefers-color-scheme: dark)'

function systemTheme(): ResolvedTheme {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return 'dark'
  return window.matchMedia(MEDIA_QUERY).matches ? 'dark' : 'light'
}

export function resolveTheme(preference: ThemePreference): ResolvedTheme {
  return preference === 'system' ? systemTheme() : preference
}

function applyTheme(preference: ThemePreference): ResolvedTheme {
  const resolved = resolveTheme(preference)
  if (typeof document !== 'undefined') {
    document.documentElement.classList.toggle('dark', resolved === 'dark')
    document.documentElement.style.colorScheme = resolved
  }
  return resolved
}

interface ThemeState {
  preference: ThemePreference
  resolved: ResolvedTheme
  setPreference: (preference: ThemePreference) => void
  /** Cycles between the two explicit themes; used by the top-bar toggle. */
  toggleTheme: () => void
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      // Dark is this product's primary theme, so it is the default rather
      // than deferring to the OS on first run.
      preference: 'dark',
      resolved: 'dark',
      setPreference: (preference) => {
        set({ preference, resolved: applyTheme(preference) })
      },
      toggleTheme: () => {
        get().setPreference(get().resolved === 'dark' ? 'light' : 'dark')
      },
    }),
    {
      name: 'pixelforge.theme',
      partialize: (state) => ({ preference: state.preference }),
      onRehydrateStorage: () => (state) => {
        const preference = state?.preference ?? 'dark'
        useThemeStore.setState({ resolved: applyTheme(preference) })
      },
    },
  ),
)

/**
 * Apply the stored theme and start following the OS preference.
 * Called once from main.tsx before first paint to avoid a flash of the wrong theme.
 */
export function initTheme(): () => void {
  applyTheme(useThemeStore.getState().preference)

  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return () => undefined
  }

  const mediaQueryList = window.matchMedia(MEDIA_QUERY)
  const onChange = () => {
    // Only react when the user is actually deferring to the system.
    if (useThemeStore.getState().preference === 'system') {
      useThemeStore.setState({ resolved: applyTheme('system') })
    }
  }

  mediaQueryList.addEventListener('change', onChange)
  return () => { mediaQueryList.removeEventListener('change', onChange) }
}
