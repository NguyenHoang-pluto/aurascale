import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { useZoomPan } from './useZoomPan'

describe('useZoomPan', () => {
  /**
   * Regression guard.
   *
   * The hook previously took a `{ width, height }` object. Callers built it
   * inline, so it was a new reference on every render; the size-dependent
   * effect then re-ran every render and set fresh transform state each time,
   * looping forever and hanging the tab. Taking primitives makes that
   * impossible, and this asserts the resulting stability.
   */
  it('returns a stable transform when re-rendered with unchanged dimensions', () => {
    const { result, rerender } = renderHook(
      ({ width, height }: { width: number; height: number }) => useZoomPan(width, height),
      { initialProps: { width: 1600, height: 1200 } },
    )

    const first = result.current.transform
    rerender({ width: 1600, height: 1200 })
    rerender({ width: 1600, height: 1200 })

    expect(result.current.transform).toBe(first)
  })

  it('starts fitted at an identity transform before the container is measured', () => {
    const { result } = renderHook(() => useZoomPan(1600, 1200))

    expect(result.current.isFitted).toBe(true)
    expect(result.current.transform).toEqual({ scale: 1, tx: 0, ty: 0 })
    // With no measured container there is nowhere to pan to.
    expect(result.current.canPan).toBe(false)
  })

  it('leaves the fitted state when a zoom is chosen and returns on fit', () => {
    const { result } = renderHook(() => useZoomPan(1600, 1200))

    act(() => { result.current.setScale(2) })
    expect(result.current.isFitted).toBe(false)
    expect(result.current.transform.scale).toBe(2)

    act(() => { result.current.fit() })
    expect(result.current.isFitted).toBe(true)
  })

  it('does not step above the largest or below the smallest preset', () => {
    const { result } = renderHook(() => useZoomPan(1600, 1200))

    act(() => { result.current.setScale(4) })
    act(() => { result.current.zoomIn() })
    expect(result.current.transform.scale).toBe(4)

    act(() => { result.current.setScale(0.25) })
    act(() => { result.current.zoomOut() })
    expect(result.current.transform.scale).toBe(0.25)
  })
})
