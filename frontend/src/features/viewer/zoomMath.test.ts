import { describe, expect, it } from 'vitest'
import {
  canPan,
  clampScale,
  clampTranslation,
  computeFitScale,
  fitTransform,
  MAX_SCALE,
  MIN_SCALE,
  nextPresetDown,
  nextPresetUp,
  zoomAtPoint,
  zoomToScale,
} from './zoomMath'

const CONTAINER = { width: 800, height: 600 }

describe('computeFitScale', () => {
  it('fits a wide image by its width', () => {
    expect(computeFitScale(CONTAINER, { width: 1600, height: 600 })).toBe(0.5)
  })

  it('fits a tall image by its height', () => {
    expect(computeFitScale(CONTAINER, { width: 800, height: 2400 })).toBe(0.25)
  })

  it('never enlarges past actual size', () => {
    expect(computeFitScale(CONTAINER, { width: 100, height: 100 })).toBe(1)
  })

  it('returns a safe value before the container has been measured', () => {
    expect(computeFitScale({ width: 0, height: 0 }, { width: 100, height: 100 })).toBe(1)
    expect(computeFitScale(CONTAINER, { width: 0, height: 0 })).toBe(1)
  })
})

describe('clampTranslation', () => {
  it('centres an image smaller than the container on both axes', () => {
    const result = clampTranslation(
      { scale: 1, tx: 999, ty: -999 },
      CONTAINER,
      { width: 400, height: 300 },
    )

    expect(result.tx).toBe(200)
    expect(result.ty).toBe(150)
  })

  it('stops a larger image from being dragged past its leading edge', () => {
    const result = clampTranslation(
      { scale: 1, tx: 50, ty: 30 },
      CONTAINER,
      { width: 1600, height: 1200 },
    )

    expect(result.tx).toBe(0)
    expect(result.ty).toBe(0)
  })

  it('stops a larger image from being dragged past its trailing edge', () => {
    const result = clampTranslation(
      { scale: 1, tx: -5000, ty: -5000 },
      CONTAINER,
      { width: 1600, height: 1200 },
    )

    expect(result.tx).toBe(800 - 1600)
    expect(result.ty).toBe(600 - 1200)
  })

  it('centres one axis while clamping the other', () => {
    const result = clampTranslation(
      { scale: 1, tx: -100, ty: 400 },
      CONTAINER,
      { width: 1600, height: 200 },
    )

    expect(result.tx).toBe(-100)
    expect(result.ty).toBe(200)
  })
})

describe('fitTransform', () => {
  it('scales to fit and centres the result', () => {
    const result = fitTransform(CONTAINER, { width: 1600, height: 600 })

    expect(result.scale).toBe(0.5)
    expect(result.tx).toBe(0)
    expect(result.ty).toBe(150)
  })
})

describe('zoomAtPoint', () => {
  it('keeps the anchored point stationary', () => {
    const image = { width: 4000, height: 3000 }
    const start = { scale: 1, tx: -100, ty: -50 }
    const anchor = { x: 300, y: 200 }

    // The image coordinate under the anchor before zooming...
    const imageX = (anchor.x - start.tx) / start.scale
    const imageY = (anchor.y - start.ty) / start.scale

    const zoomed = zoomAtPoint(start, 2, anchor, CONTAINER, image)

    // ...must still be under the anchor afterwards.
    expect(zoomed.tx + imageX * zoomed.scale).toBeCloseTo(anchor.x, 5)
    expect(zoomed.ty + imageY * zoomed.scale).toBeCloseTo(anchor.y, 5)
  })

  it('clamps the scale to the supported range', () => {
    const image = { width: 4000, height: 3000 }
    const anchor = { x: 0, y: 0 }

    expect(zoomAtPoint({ scale: 1, tx: 0, ty: 0 }, 500, anchor, CONTAINER, image).scale).toBe(
      MAX_SCALE,
    )
    expect(
      zoomAtPoint({ scale: 1, tx: 0, ty: 0 }, 0.0001, anchor, CONTAINER, image).scale,
    ).toBe(MIN_SCALE)
  })
})

describe('zoomToScale', () => {
  it('keeps a zoomed-out image centred', () => {
    const result = zoomToScale(
      { scale: 1, tx: 0, ty: 0 },
      0.25,
      CONTAINER,
      { width: 1600, height: 1200 },
    )

    expect(result.scale).toBe(0.25)
    expect(result.tx).toBe((800 - 400) / 2)
    expect(result.ty).toBe((600 - 300) / 2)
  })
})

describe('preset navigation', () => {
  it('steps up through the presets and stops at the top', () => {
    expect(nextPresetUp(0.25)).toBe(0.5)
    expect(nextPresetUp(1)).toBe(2)
    expect(nextPresetUp(4)).toBeNull()
  })

  it('steps down through the presets and stops at the bottom', () => {
    expect(nextPresetDown(4)).toBe(2)
    expect(nextPresetDown(1)).toBe(0.5)
    expect(nextPresetDown(0.25)).toBeNull()
  })

  it('moves off a non-preset scale in both directions', () => {
    expect(nextPresetUp(0.73)).toBe(1)
    expect(nextPresetDown(0.73)).toBe(0.5)
  })
})

describe('canPan', () => {
  it('is false when the image fits entirely', () => {
    expect(canPan({ scale: 0.5, tx: 0, ty: 0 }, CONTAINER, { width: 1600, height: 1200 })).toBe(
      false,
    )
  })

  it('is false before the container has been measured', () => {
    expect(
      canPan({ scale: 1, tx: 0, ty: 0 }, { width: 0, height: 0 }, { width: 1600, height: 1200 }),
    ).toBe(false)
  })

  it('is true when either axis overflows', () => {
    expect(canPan({ scale: 1, tx: 0, ty: 0 }, CONTAINER, { width: 1600, height: 100 })).toBe(true)
    expect(canPan({ scale: 1, tx: 0, ty: 0 }, CONTAINER, { width: 100, height: 1200 })).toBe(true)
  })
})

describe('clampScale', () => {
  it('bounds the scale to the supported range', () => {
    expect(clampScale(100)).toBe(MAX_SCALE)
    expect(clampScale(0)).toBe(MIN_SCALE)
    expect(clampScale(1.5)).toBe(1.5)
  })
})
