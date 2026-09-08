import { describe, expect, it } from 'vitest'
import {
  DIVIDER_COARSE_STEP,
  DIVIDER_STEP,
  MAX_CROP_EDGE,
  PREVIEW_MAX_EDGE,
  previewDetailScale,
  MAX_DIVIDER,
  MIN_DIVIDER,
  SPLIT_POSITION,
  clampDivider,
  cropForView,
  dividerForKey,
  dividerFromPointer,
  sameCrop,
  visibleRegion,
} from './comparisonMath'

const CONTAINER = { width: 800, height: 600 }
const IMAGE = { width: 4000, height: 3000 }

function rect(left: number, width: number): DOMRect {
  return { left, width, top: 0, height: 0, right: left + width, bottom: 0, x: left, y: 0,
    toJSON: () => ({}) } as DOMRect
}

// ------------------------------------------------------------------ divider

describe('clampDivider', () => {
  it('keeps a position inside the frame', () => {
    expect(clampDivider(-20)).toBe(MIN_DIVIDER)
    expect(clampDivider(140)).toBe(MAX_DIVIDER)
    expect(clampDivider(37)).toBe(37)
  })

  it('falls back to the middle for a value that is not a number', () => {
    expect(clampDivider(Number.NaN)).toBe(SPLIT_POSITION)
  })

  it('allows both extremes, so either image can be seen whole', () => {
    expect(clampDivider(0)).toBe(0)
    expect(clampDivider(100)).toBe(100)
  })
})

describe('dividerFromPointer', () => {
  it('maps a pointer position to a percentage of the surface', () => {
    expect(dividerFromPointer(500, rect(100, 800))).toBe(50)
  })

  it('clamps a pointer dragged outside the surface', () => {
    expect(dividerFromPointer(0, rect(100, 800))).toBe(MIN_DIVIDER)
    expect(dividerFromPointer(5000, rect(100, 800))).toBe(MAX_DIVIDER)
  })

  it('does not divide by a zero width', () => {
    expect(dividerFromPointer(50, rect(0, 0))).toBe(SPLIT_POSITION)
  })
})

describe('dividerForKey', () => {
  it('steps with the arrow keys', () => {
    expect(dividerForKey('ArrowLeft', 50)).toBe(50 - DIVIDER_STEP)
    expect(dividerForKey('ArrowRight', 50)).toBe(50 + DIVIDER_STEP)
  })

  it('takes larger steps with shift held', () => {
    expect(dividerForKey('ArrowRight', 50, { coarse: true })).toBe(50 + DIVIDER_COARSE_STEP)
  })

  it('jumps to the edges with Home and End', () => {
    expect(dividerForKey('Home', 50)).toBe(MIN_DIVIDER)
    expect(dividerForKey('End', 50)).toBe(MAX_DIVIDER)
  })

  it('clamps rather than running past an edge', () => {
    expect(dividerForKey('ArrowLeft', 1)).toBe(MIN_DIVIDER)
    expect(dividerForKey('ArrowRight', 99)).toBe(MAX_DIVIDER)
  })

  it('leaves keys it does not own alone', () => {
    // The surface pans with the arrows too; anything else must fall through.
    expect(dividerForKey('ArrowUp', 50)).toBeNull()
    expect(dividerForKey('a', 50)).toBeNull()
  })
})

// ----------------------------------------------------------- visible region

describe('visibleRegion', () => {
  it('is the whole image when it is fitted inside the container', () => {
    // 800/4000 = 0.2 fits the image exactly across.
    const region = visibleRegion({ scale: 0.2, tx: 0, ty: 0 }, CONTAINER, IMAGE)

    expect(region).toEqual({ x: 0, y: 0, width: 4000, height: 3000 })
  })

  it('is the part on screen when zoomed in', () => {
    // At 2x with no offset, the container shows 400x300 image pixels.
    const region = visibleRegion({ scale: 2, tx: 0, ty: 0 }, CONTAINER, IMAGE)

    expect(region).toEqual({ x: 0, y: 0, width: 400, height: 300 })
  })

  it('follows the pan', () => {
    const region = visibleRegion({ scale: 2, tx: -1000, ty: -600 }, CONTAINER, IMAGE)

    expect(region.x).toBe(500)
    expect(region.y).toBe(300)
  })

  it('never reports pixels outside the image', () => {
    const region = visibleRegion({ scale: 2, tx: 200, ty: 200 }, CONTAINER, IMAGE)

    expect(region.x).toBe(0)
    expect(region.y).toBe(0)
  })

  it('is empty before the container has been measured', () => {
    expect(visibleRegion({ scale: 1, tx: 0, ty: 0 }, { width: 0, height: 0 }, IMAGE)).toEqual({
      x: 0,
      y: 0,
      width: 0,
      height: 0,
    })
  })
})

// -------------------------------------------------------------- crop policy

describe('cropForView', () => {
  it('asks for nothing at or below 100%', () => {
    // This result is inside the preview cap, so the preview is 1:1 with the
    // output and still has more detail than the screen can show.
    expect(cropForView({ scale: 1, tx: 0, ty: 0 }, CONTAINER, IMAGE)).toBeNull()
    expect(cropForView({ scale: 0.5, tx: 0, ty: 0 }, CONTAINER, IMAGE)).toBeNull()
  })

  it('asks for the visible region above 100%', () => {
    const crop = cropForView({ scale: 2, tx: 0, ty: 0 }, CONTAINER, IMAGE)

    expect(crop).toEqual({ x: 0, y: 0, w: 400, h: 300 })
  })

  it('declines a region larger than the cap', () => {
    // Barely above 1x on a huge container would ask for most of the image.
    const wide = { width: MAX_CROP_EDGE * 2, height: MAX_CROP_EDGE * 2 }

    expect(cropForView({ scale: 1.01, tx: 0, ty: 0 }, wide, IMAGE)).toBeNull()
  })

  it('asks for nothing when there is no visible area', () => {
    expect(cropForView({ scale: 4, tx: 0, ty: 0 }, { width: 0, height: 0 }, IMAGE)).toBeNull()
  })
})

describe('previewDetailScale', () => {
  it('is 1:1 while the result fits inside the preview cap', () => {
    expect(previewDetailScale({ width: 4000, height: 3000 })).toBe(1)
    expect(previewDetailScale({ width: PREVIEW_MAX_EDGE, height: 100 })).toBe(1)
  })

  it('falls as the result outgrows the cap', () => {
    // A 16000 px 8x result is served as a 4096 px preview.
    expect(previewDetailScale({ width: 16000, height: 12000 })).toBeCloseTo(0.256, 5)
    expect(previewDetailScale({ width: 8000, height: 6000 })).toBeCloseTo(0.512, 5)
  })

  it('is measured on the long edge, whichever it is', () => {
    expect(previewDetailScale({ width: 12000, height: 16000 })).toBeCloseTo(0.256, 5)
  })

  it('does not divide by a degenerate size', () => {
    expect(previewDetailScale({ width: 0, height: 0 })).toBe(1)
  })
})

// The reported case: 2000x1500 at 8x.
const HUGE = { width: 16000, height: 12000 }

describe('cropForView for a result larger than the preview cap', () => {
  it('fetches real pixels at 100%, where the preview is only a quarter', () => {
    // The regression. The old rule was `scale <= 1`, so at exactly 100% this
    // returned null and the viewer stretched a 4096 px JPEG by four.
    const crop = cropForView({ scale: 1, tx: 0, ty: 0 }, CONTAINER, HUGE)

    expect(crop).not.toBeNull()
    expect(crop).toEqual({ x: 0, y: 0, w: 800, h: 600 })
  })

  it('fetches real pixels at 200% and 400% too', () => {
    for (const scale of [2, 4]) {
      expect(cropForView({ scale, tx: 0, ty: 0 }, CONTAINER, HUGE)).not.toBeNull()
    }
  })

  it('still uses the preview at fit-to-screen', () => {
    // Fitting 16000 px into an 800 px container is ~0.05x - far below the
    // 0.256 the preview covers, so a crop would be wasted bandwidth.
    expect(cropForView({ scale: 0.05, tx: 0, ty: 0 }, CONTAINER, HUGE)).toBeNull()
    expect(cropForView({ scale: 0.2, tx: 0, ty: 0 }, CONTAINER, HUGE)).toBeNull()
  })

  it('switches over exactly where the preview runs out', () => {
    const threshold = previewDetailScale(HUGE)
    // A small viewport, so the size cap cannot gate the result first and the
    // threshold itself is what is under test. The two rules are independent:
    // at 0.27x on an 800 px container the region is ~2976 px and the cap
    // declines it regardless, which the next test covers.
    const small = { width: 512, height: 512 }

    expect(cropForView({ scale: threshold, tx: 0, ty: 0 }, small, HUGE)).toBeNull()
    expect(cropForView({ scale: threshold * 1.05, tx: 0, ty: 0 }, small, HUGE)).not.toBeNull()
  })

  it('never asks for more than the cap, however far out the user zooms', () => {
    // The bound that stops a 16K result becoming a 16K request.
    const container = { width: 6000, height: 6000 }

    for (const scale of [0.3, 0.5, 0.9, 1, 2, 4]) {
      const crop = cropForView({ scale, tx: 0, ty: 0 }, container, HUGE)
      if (crop === null) continue
      expect(crop.w).toBeLessThanOrEqual(MAX_CROP_EDGE)
      expect(crop.h).toBeLessThanOrEqual(MAX_CROP_EDGE)
    }
  })

  it('leaves a result inside the cap behaving exactly as before', () => {
    // No behaviour change for 2x and 4x results, which is most of them.
    expect(cropForView({ scale: 1, tx: 0, ty: 0 }, CONTAINER, IMAGE)).toBeNull()
    expect(cropForView({ scale: 1.5, tx: 0, ty: 0 }, CONTAINER, IMAGE)).not.toBeNull()
  })
})

describe('sameCrop', () => {
  it('recognises an unchanged region so it is not refetched', () => {
    const region = { x: 1, y: 2, w: 3, h: 4 }

    expect(sameCrop(region, { ...region })).toBe(true)
  })

  it('spots a moved or resized region', () => {
    expect(sameCrop({ x: 1, y: 2, w: 3, h: 4 }, { x: 9, y: 2, w: 3, h: 4 })).toBe(false)
    expect(sameCrop({ x: 1, y: 2, w: 3, h: 4 }, { x: 1, y: 2, w: 30, h: 4 })).toBe(false)
  })

  it('handles the absence of a region on either side', () => {
    expect(sameCrop(null, null)).toBe(true)
    expect(sameCrop(null, { x: 0, y: 0, w: 1, h: 1 })).toBe(false)
    expect(sameCrop({ x: 0, y: 0, w: 1, h: 1 }, null)).toBe(false)
  })
})
