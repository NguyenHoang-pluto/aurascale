import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export type Theme = 'dark' | 'light'

interface ThemeState {
  theme: Theme
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
}

/** Apply the theme class the Tailwind `dark` variant keys off. */
function applyTheme(theme: Theme): void {
  if (typeof document === 'undefined') return
  document.documentElement.classList.toggle('dark', theme === 'dark')
  document.documentElement.style.colorScheme = theme
}

export const useThemeStore = create<ThemeState>()(
  persist(
    (set, get) => ({
      // Dark is the primary theme for this product, so it is the default
      // rather than following the OS preference.
      theme: 'dark',
      setTheme: (theme) => {
        applyTheme(theme)
        set({ theme })
      },
      toggleTheme: () => {
        get().setTheme(get().theme === 'dark' ? 'light' : 'dark')
      },
    }),
    {
      name: 'pixelforge.theme',
      onRehydrateStorage: () => (state) => {
        applyTheme(state?.theme ?? 'dark')
      },
    },
  ),
)

/** Called once from main.tsx before first paint to avoid a theme flash. */
export function initTheme(): void {
  applyTheme(useThemeStore.getState().theme)
}
