import { useCallback, useSyncExternalStore } from 'react'

function isSupported(): boolean {
  return typeof window !== 'undefined' && typeof window.matchMedia === 'function'
}

/**
 * Subscribe to a CSS media query.
 *
 * Implemented with `useSyncExternalStore` rather than `useState` + `useEffect`:
 * the media query list is an external store, and this is the API built for
 * that. It reads the current value during render, so there is no flash of the
 * wrong layout on mount and no setState-in-effect render cascade.
 *
 * Used for layout decisions that cannot be expressed in CSS alone — rendering
 * a different navigation component rather than hiding one, so the component's
 * contract does not depend on a stylesheet having loaded.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onStoreChange: () => void) => {
      if (!isSupported()) return () => undefined

      const mediaQueryList = window.matchMedia(query)
      mediaQueryList.addEventListener('change', onStoreChange)
      return () => { mediaQueryList.removeEventListener('change', onStoreChange) }
    },
    [query],
  )

  const getSnapshot = useCallback(() => {
    if (!isSupported()) return false
    return window.matchMedia(query).matches
  }, [query])

  // Server rendering has no viewport; assume the desktop layout.
  const getServerSnapshot = useCallback(() => false, [])

  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}

/** Tailwind's `md` breakpoint. Below this the workspace stacks vertically. */
export const MOBILE_QUERY = '(max-width: 767px)'
