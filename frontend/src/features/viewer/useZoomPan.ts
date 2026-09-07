import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import {
  canPan as canPanFor,
  clampTranslation,
  computeFitScale,
  fitTransform,
  nextPresetDown,
  nextPresetUp,
  transformsEqual,
  zoomAtPoint,
  zoomToScale,
  type Size,
  type Transform,
} from './zoomMath'

const IDENTITY: Transform = { scale: 1, tx: 0, ty: 0 }

export interface UseZoomPanResult {
  /** Callback ref: attach to the element that clips the image. */
  attachContainer: (node: HTMLElement | null) => void
  transform: Transform
  /** True while the view is tracking the fit scale rather than a chosen zoom. */
  isFitted: boolean
  isPanning: boolean
  canPan: boolean
  fitScale: number
  zoomIn: () => void
  zoomOut: () => void
  setScale: (scale: number) => void
  fit: () => void
  actualSize: () => void
  handlers: {
    onPointerDown: (event: React.PointerEvent<HTMLElement>) => void
    onPointerMove: (event: React.PointerEvent<HTMLElement>) => void
    onPointerUp: (event: React.PointerEvent<HTMLElement>) => void
    onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => void
  }
}

/**
 * Zoom and pan state for the image viewer.
 *
 * Sizing comes from a ResizeObserver rather than a one-off measurement, so the
 * fitted view stays correct when the settings rail collapses or the window is
 * resized. While the view is "fitted" it re-fits on resize; once the user
 * chooses a zoom, resizing no longer overrides their choice.
 *
 * Takes the dimensions as two numbers rather than a `Size` object on purpose:
 * an object argument would be a fresh reference on every render, which makes
 * the size-dependent effect below re-run every render and set state each time
 * -- an infinite render loop that hangs the tab.
 */
export function useZoomPan(imageWidth: number, imageHeight: number): UseZoomPanResult {
  const image = useMemo<Size>(
    () => ({ width: imageWidth, height: imageHeight }),
    [imageWidth, imageHeight],
  )

  const [container, setContainer] = useState<Size>({ width: 0, height: 0 })
  const [transform, setTransform] = useState<Transform>(IDENTITY)
  const [isFitted, setIsFitted] = useState(true)
  const [isPanning, setIsPanning] = useState(false)

  const nodeRef = useRef<HTMLElement | null>(null)
  const panOrigin = useRef<{ x: number; y: number; tx: number; ty: number } | null>(null)
  // Latest state, read inside event handlers that must keep a stable identity
  // (the wheel listener is bound natively and should not be re-subscribed on
  // every zoom). Synced in an effect rather than during render: mutating a ref
  // while rendering is unsafe under concurrent rendering, and events only fire
  // after commit, so the handlers always observe current values.
  const stateRef = useRef({ container, transform, isFitted, image })
  useEffect(() => {
    stateRef.current = { container, transform, isFitted, image }
  })

  const attachContainer = useCallback((node: HTMLElement | null) => {
    nodeRef.current = node
    if (node !== null) {
      const rect = node.getBoundingClientRect()
      setContainer({ width: rect.width, height: rect.height })
    }
  }, [])

  useLayoutEffect(() => {
    const node = nodeRef.current
    if (node === null || typeof ResizeObserver === 'undefined') return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry === undefined) return
      const { width, height } = entry.contentRect
      setContainer((previous) =>
        previous.width === width && previous.height === height
          ? previous
          : { width, height },
      )
    })
    observer.observe(node)
    return () => { observer.disconnect() }
  }, [])

  // Re-fit when the image changes, or when the container resizes while fitted.
  useEffect(() => {
    if (container.width <= 0 || image.width <= 0) return
    // Every update goes through an equality check so an unchanged result never
    // schedules another render.
    const update = (next: (current: Transform) => Transform) => {
      setTransform((current) => {
        const candidate = next(current)
        return transformsEqual(current, candidate) ? current : candidate
      })
    }

    if (!isFitted) {
      // Keep a manually chosen zoom, but re-clamp so a resize cannot leave the
      // image stranded outside the viewport.
      update((current) => clampTranslation(current, container, image))
      return
    }
    update(() => fitTransform(container, image))
  }, [container, image, isFitted])

  const apply = useCallback((next: Transform, fitted: boolean) => {
    setTransform(next)
    setIsFitted(fitted)
  }, [])

  const setScale = useCallback(
    (scale: number) => {
      const { container: c, transform: t, image: i } = stateRef.current
      apply(zoomToScale(t, scale, c, i), false)
    },
    [apply],
  )

  const zoomIn = useCallback(() => {
    const next = nextPresetUp(stateRef.current.transform.scale)
    if (next !== null) setScale(next)
  }, [setScale])

  const zoomOut = useCallback(() => {
    const next = nextPresetDown(stateRef.current.transform.scale)
    if (next !== null) setScale(next)
  }, [setScale])

  const fit = useCallback(() => {
    const { container: c, image: i } = stateRef.current
    apply(fitTransform(c, i), true)
  }, [apply])

  const actualSize = useCallback(() => { setScale(1) }, [setScale])

  // Wheel is bound natively rather than via React's onWheel: React attaches a
  // passive listener, which cannot preventDefault, and the browser would zoom
  // the whole page on ctrl+wheel.
  useEffect(() => {
    const node = nodeRef.current
    if (node === null) return

    const onWheel = (event: WheelEvent) => {
      const { container: c, transform: t, image: i } = stateRef.current
      if (i.width <= 0) return

      // Ctrl/Cmd+wheel is the pinch-zoom gesture on trackpads; a plain wheel
      // scrolls the page, which is the expected default.
      if (!event.ctrlKey && !event.metaKey) return

      event.preventDefault()
      const rect = node.getBoundingClientRect()
      const anchor = { x: event.clientX - rect.left, y: event.clientY - rect.top }
      const factor = Math.exp(-event.deltaY / 300)
      apply(zoomAtPoint(t, t.scale * factor, anchor, c, i), false)
    }

    node.addEventListener('wheel', onWheel, { passive: false })
    return () => { node.removeEventListener('wheel', onWheel) }
  }, [apply])

  const onPointerDown = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const { container: c, transform: t, image: i } = stateRef.current
    if (!canPanFor(t, c, i) || event.button !== 0) return

    // Guarded: pointer capture keeps a drag alive when the cursor leaves the
    // element, but it is not implemented everywhere, and losing it should
    // degrade the drag rather than break it.
    if (typeof event.currentTarget.setPointerCapture === 'function') {
      event.currentTarget.setPointerCapture(event.pointerId)
    }
    panOrigin.current = { x: event.clientX, y: event.clientY, tx: t.tx, ty: t.ty }
    setIsPanning(true)
  }, [])

  const onPointerMove = useCallback((event: React.PointerEvent<HTMLElement>) => {
    const origin = panOrigin.current
    if (origin === null) return

    const { container: c, image: i } = stateRef.current
    setTransform((current) =>
      clampTranslation(
        {
          scale: current.scale,
          tx: origin.tx + (event.clientX - origin.x),
          ty: origin.ty + (event.clientY - origin.y),
        },
        c,
        i,
      ),
    )
  }, [])

  const onPointerUp = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (panOrigin.current === null) return
    if (
      typeof event.currentTarget.hasPointerCapture === 'function' &&
      event.currentTarget.hasPointerCapture(event.pointerId)
    ) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    panOrigin.current = null
    setIsPanning(false)
  }, [])

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLElement>) => {
      const { container: c, transform: t, image: i } = stateRef.current
      const step = event.shiftKey ? 100 : 25
      const pan = (dx: number, dy: number) => {
        event.preventDefault()
        apply(
          clampTranslation({ scale: t.scale, tx: t.tx + dx, ty: t.ty + dy }, c, i),
          false,
        )
      }

      switch (event.key) {
        case '+':
        case '=':
          event.preventDefault()
          zoomIn()
          break
        case '-':
        case '_':
          event.preventDefault()
          zoomOut()
          break
        case '0':
          event.preventDefault()
          fit()
          break
        case '1':
          event.preventDefault()
          actualSize()
          break
        case 'ArrowLeft':
          pan(step, 0)
          break
        case 'ArrowRight':
          pan(-step, 0)
          break
        case 'ArrowUp':
          pan(0, step)
          break
        case 'ArrowDown':
          pan(0, -step)
          break
        default:
          break
      }
    },
    [actualSize, apply, fit, zoomIn, zoomOut],
  )

  return {
    attachContainer,
    transform,
    isFitted,
    isPanning,
    canPan: canPanFor(transform, container, image),
    fitScale: computeFitScale(container, image),
    zoomIn,
    zoomOut,
    setScale,
    fit,
    actualSize,
    handlers: { onPointerDown, onPointerMove, onPointerUp, onKeyDown },
  }
}
