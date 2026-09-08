import { create } from 'zustand'
import { persist } from 'zustand/middleware'

/**
 * The interface language.
 *
 * Deliberately mirrors `useThemeStore`: the same `persist` middleware, the
 * same "apply before first paint" entry point called from `main.tsx`. That is
 * also why this project has no language-detector plugin — the mechanism one
 * would provide already exists here, and a second copy of it could disagree
 * with this one about what the user last chose.
 */

export const LANGUAGES = ['en', 'vi'] as const
export type Language = (typeof LANGUAGES)[number]

/** English is the default and the fallback for anything untranslated. */
export const DEFAULT_LANGUAGE: Language = 'en'

export const STORAGE_KEY = 'pixelforge.language'

export function isLanguage(value: unknown): value is Language {
  return typeof value === 'string' && (LANGUAGES as readonly string[]).includes(value)
}

/**
 * What the browser is asking for, if anything usable.
 *
 * Only the primary subtag is considered: `vi-VN` and `vi` are the same choice
 * as far as this application is concerned.
 */
export function detectBrowserLanguage(
  navigatorLanguages: readonly string[] | undefined,
): Language | null {
  for (const tag of navigatorLanguages ?? []) {
    const primary = tag.toLowerCase().split('-')[0]
    if (isLanguage(primary)) return primary
  }
  return null
}

interface LanguageState {
  language: Language
  setLanguage: (language: Language) => void
}

export const useLanguageStore = create<LanguageState>()(
  persist(
    (set) => ({
      language: DEFAULT_LANGUAGE,
      setLanguage: (language) => { set({ language }) },
    }),
    {
      name: STORAGE_KEY,
      partialize: (state) => ({ language: state.language }),
    },
  ),
)

/**
 * The language to start in: what was stored, else what the browser asks for,
 * else English.
 *
 * `hasHydrated` is not consulted because `persist` rehydrates synchronously
 * from `localStorage`; by the time this is called from `main.tsx` the stored
 * value is already in the store.
 */
export function resolveInitialLanguage(): Language {
  const stored = useLanguageStore.getState().language
  // A stored value only counts if the user actually chose it. On a first run
  // the store holds the default, which must not mask the browser preference.
  if (readStoredLanguage() !== null) return stored

  const navigatorLanguages =
    typeof navigator === 'undefined'
      ? undefined
      : (navigator.languages ?? [navigator.language])

  return detectBrowserLanguage(navigatorLanguages) ?? DEFAULT_LANGUAGE
}

/** The persisted choice, or null when the user has never made one. */
export function readStoredLanguage(): Language | null {
  if (typeof localStorage === 'undefined') return null

  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw === null) return null

    const parsed: unknown = JSON.parse(raw)
    const candidate = (parsed as { state?: { language?: unknown } }).state?.language
    return isLanguage(candidate) ? candidate : null
  } catch {
    // A corrupt or inaccessible store is not worth failing over; the browser
    // preference and the default are both still available.
    return null
  }
}
