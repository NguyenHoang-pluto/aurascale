import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { initTheme, resolveTheme, useThemeStore } from './useThemeStore'

type MediaListener = (event: MediaQueryListEvent) => void

/** Minimal matchMedia stub that can flip the system preference on demand. */
function stubMatchMedia(prefersDark: boolean) {
  const listeners = new Set<MediaListener>()
  const mql = {
    matches: prefersDark,
    media: '(prefers-color-scheme: dark)',
    addEventListener: (_: string, listener: MediaListener) => { listeners.add(listener) },
    removeEventListener: (_: string, listener: MediaListener) => { listeners.delete(listener) },
  }
  vi.stubGlobal(
    'matchMedia',
    vi.fn(() => mql),
  )
  return {
    setPrefersDark(next: boolean) {
      mql.matches = next
      for (const listener of listeners) {
        listener({ matches: next } as MediaQueryListEvent)
      }
    },
    listenerCount: () => listeners.size,
  }
}

describe('useThemeStore', () => {
  beforeEach(() => {
    useThemeStore.setState({ preference: 'dark', resolved: 'dark' })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('defaults to dark, this product having a dark primary theme', () => {
    expect(useThemeStore.getState().preference).toBe('dark')
  })

  it('applies the dark class and color-scheme to the document element', () => {
    useThemeStore.getState().setPreference('light')

    expect(document.documentElement.classList.contains('dark')).toBe(false)
    expect(document.documentElement.style.colorScheme).toBe('light')

    useThemeStore.getState().setPreference('dark')

    expect(document.documentElement.classList.contains('dark')).toBe(true)
    expect(document.documentElement.style.colorScheme).toBe('dark')
  })

  it('resolves the system preference to a concrete theme', () => {
    stubMatchMedia(false)
    expect(resolveTheme('system')).toBe('light')

    stubMatchMedia(true)
    expect(resolveTheme('system')).toBe('dark')
  })

  it('toggles between the two explicit themes from whatever is resolved', () => {
    stubMatchMedia(true)
    useThemeStore.getState().setPreference('system')
    expect(useThemeStore.getState().resolved).toBe('dark')

    useThemeStore.getState().toggleTheme()

    expect(useThemeStore.getState().preference).toBe('light')
    expect(useThemeStore.getState().resolved).toBe('light')
  })

  it('follows OS changes only while the preference is "system"', () => {
    const media = stubMatchMedia(true)
    const cleanup = initTheme()

    useThemeStore.getState().setPreference('system')
    media.setPrefersDark(false)
    expect(useThemeStore.getState().resolved).toBe('light')

    // An explicit choice must not be overridden by the OS.
    useThemeStore.getState().setPreference('dark')
    media.setPrefersDark(false)
    expect(useThemeStore.getState().resolved).toBe('dark')

    cleanup()
    expect(media.listenerCount()).toBe(0)
  })
})
