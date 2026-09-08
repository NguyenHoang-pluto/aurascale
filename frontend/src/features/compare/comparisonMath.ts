import type { Size, Transform } from '@/features/viewer/zoomMath'

/**
 * Geometry for the before/after comparison.
 *
 * The comparison works in **output pixel space**: the enhanced result defines
 * the coordinate system, and the original is drawn stretched into the same
 * frame. That is what makes a wipe honest — at any divider position the two
 * halves show the same region of the same subject, which is the whole point of
 * putting them side by side.
 *
 * The shared transform is therefore the output image's transform, and nothing
 * here changes its meaning.
 */

export const MIN_DIVIDER = 0
export const MAX_DIVIDER = 100
/** Arrow-key step, in percent. Shift multiplies it. */
export const DIVIDER_STEP = 2
export const DIVIDER_COARSE_STEP = 10

export type CompareMode = 'slider' | 'side-by-side' | 'split'

/** Where the fixed split sits. Not adjustable, by definition. */
export const SPLIT_POSITION = 50

export function clampDivider(position: number): number {
  if (!Number.isFinite(position)) return SPLIT_POSITION
  return Math.min(MAX_DIVIDER, Math.max(MIN_DIVIDER, position))
}

/** The divider position a pointer at `clientX` implies, as a percentage. */
export function dividerFromPointer(clientX: number, bounds: DOMRect): number {
  if (bounds.width <= 0) return SPLIT_POSITION
  return clampDivider(((clientX - bounds.left) / bounds.width) * 100)
}

/** The next divider position for a key, or null when the key is not ours. */
export function dividerForKey(
  key: string,
  current: number,
  options: { coarse?: boolean } = {},
): number | null {
  const step = options.coarse === true ? DIVIDER_COARSE_STEP : DIVIDER_STEP

  switch (key) {
    case 'ArrowLeft':
      return clampDivider(current - step)
    case 'ArrowRight':
      return clampDivider(current + step)
    case 'Home':
      return MIN_DIVIDER
    case 'End':
      return MAX_DIVIDER
    default:
      return null
  }
}

/**
 * The region of the output currently on screen, in output pixels.
 *
 * Inverts the transform: a pixel at container position `p` sits at
 * `(p - t) / scale` in the image. Clamped to the image, because the container
 * is usually larger than the fitted image and the surrounding space is not
 * part of the result.
 */
export function visibleRegion(
  transform: Transform,
  container: Size,
  image: Size,
): { x: number; y: number; width: number; height: number } {
  const { scale, tx, ty } = transform
  if (scale <= 0 || container.width <= 0 || image.width <= 0) {
    return { x: 0, y: 0, width: 0, height: 0 }
  }

  const left = Math.max(0, -tx / scale)
  const top = Math.max(0, -ty / scale)
  const right = Math.min(image.width, (container.width - tx) / scale)
  const bottom = Math.min(image.height, (container.height - ty) / scale)

  return {
    x: Math.floor(left),
    y: Math.floor(top),
    width: Math.max(0, Math.ceil(right - left)),
    height: Math.max(0, Math.ceil(bottom - top)),
  }
}

/** The largest region a single crop request may cover, in source pixels. */
export const MAX_CROP_EDGE = 2048

/**
 * The crop to request for what is on screen, or null when none is warranted.
 *
 * Two rules decide this. Below 100% the preview already carries more detail
 * than the screen can show, so a crop would be wasted bandwidth. And a region
 * larger than the cap is refused rather than shrunk around the centre: it
 * means the user is not zoomed in far enough for a crop to be worth fetching.
 */
export function cropForView(
  transform: Transform,
  container: Size,
  image: Size,
): { x: number; y: number; w: number; h: number } | null {
  if (transform.scale <= 1) return null

  const region = visibleRegion(transform, container, image)
  if (region.width <= 0 || region.height <= 0) return null
  if (region.width > MAX_CROP_EDGE || region.height > MAX_CROP_EDGE) return null

  return { x: region.x, y: region.y, w: region.width, h: region.height }
}

/** Whether two crops describe the same region, so a refetch can be skipped. */
export function sameCrop(
  a: { x: number; y: number; w: number; h: number } | null,
  b: { x: number; y: number; w: number; h: number } | null,
): boolean {
  if (a === null || b === null) return a === b
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h
}
