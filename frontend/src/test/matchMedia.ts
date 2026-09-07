/**
 * Deterministic `matchMedia` for tests.
 *
 * jsdom implements `matchMedia` but always reports `matches: false` and never
 * emits change events, so components that branch on a media query can only be
 * tested in one state. This mock evaluates the two query shapes the app
 * actually uses against settable state, and notifies listeners on change.
 */

interface MediaState {
  viewportWidth: number
  prefersDark: boolean
}

type Listener = (event: MediaQueryListEvent) => void

const state: MediaState = { viewportWidth: 1440, prefersDark: true }
const registry = new Map<string, Set<Listener>>()

const MAX_WIDTH = /\(max-width:\s*(\d+)px\)/
const MIN_WIDTH = /\(min-width:\s*(\d+)px\)/

function evaluate(query: string): boolean {
  if (query.includes('prefers-color-scheme: dark')) return state.prefersDark
  if (query.includes('prefers-color-scheme: light')) return !state.prefersDark

  const max = MAX_WIDTH.exec(query)
  if (max?.[1] !== undefined) return state.viewportWidth <= Number(max[1])

  const min = MIN_WIDTH.exec(query)
  if (min?.[1] !== undefined) return state.viewportWidth >= Number(min[1])

  return false
}

function notifyAll(): void {
  for (const [query, listeners] of registry) {
    const matches = evaluate(query)
    for (const listener of listeners) {
      listener({ matches, media: query } as MediaQueryListEvent)
    }
  }
}

export function installMatchMedia(): void {
  window.matchMedia = ((query: string) => {
    let listeners = registry.get(query)
    if (listeners === undefined) {
      listeners = new Set<Listener>()
      registry.set(query, listeners)
    }
    const bucket = listeners

    return {
      get matches() {
        return evaluate(query)
      },
      media: query,
      onchange: null,
      addEventListener: (_: string, listener: Listener) => { bucket.add(listener) },
      removeEventListener: (_: string, listener: Listener) => { bucket.delete(listener) },
      addListener: (listener: Listener) => { bucket.add(listener) },
      removeListener: (listener: Listener) => { bucket.delete(listener) },
      dispatchEvent: () => false,
    } as unknown as MediaQueryList
  }) as typeof window.matchMedia
}

/** Reset to the default desktop, dark-preference viewport. */
export function resetMatchMedia(): void {
  state.viewportWidth = 1440
  state.prefersDark = true
  registry.clear()
}

/** Set the simulated viewport width and notify subscribers. */
export function setViewportWidth(width: number): void {
  state.viewportWidth = width
  notifyAll()
}

/** Set the simulated OS colour-scheme preference and notify subscribers. */
export function setPrefersDark(prefersDark: boolean): void {
  state.prefersDark = prefersDark
  notifyAll()
}

/** Width used by tests that want the compact (mobile) layout. */
export const MOBILE_WIDTH = 480
