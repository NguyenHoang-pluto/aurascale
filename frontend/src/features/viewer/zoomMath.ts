/**
 * Pure geometry for the image viewer.
 *
 * Kept free of React so the arithmetic — which is where zoom/pan bugs actually
 * live — can be tested directly rather than through a rendered component.
 *
 * Coordinate model: the image is drawn at its natural size with
 * `transform: translate(tx, ty) scale(s)` and `transform-origin: 0 0`, so
 * (tx, ty) is the position of the image's top-left corner in container space.
 */

export interface Size {
  width: number
  height: number
}

export interface Transform {
  scale: number
  tx: number
  ty: number
}

/** Zoom stops offered in the UI (§ 5). */
export const ZOOM_PRESETS = [0.25, 0.5, 1, 2, 4] as const

export const MIN_SCALE = 0.05
export const MAX_SCALE = 8

export function clampScale(scale: number): number {
  return Math.min(MAX_SCALE, Math.max(MIN_SCALE, scale))
}

/**
 * Scale at which the image fits entirely within the container.
 *
 * Capped at 1: "Fit" never enlarges beyond actual size, so a small image is not
 * shown as a blurry wall of pixels. Use the zoom presets to go past 100%.
 */
export function computeFitScale(container: Size, image: Size): number {
  if (container.width <= 0 || container.height <= 0) return 1
  if (image.width <= 0 || image.height <= 0) return 1

  return Math.min(container.width / image.width, container.height / image.height, 1)
}

/**
 * Position the image for a given scale.
 *
 * An axis smaller than the container is centred; a larger one is clamped so the
 * image edge can never be dragged inside the container edge, which would leave
 * dead space and make the image feel unanchored.
 */
export function clampTranslation(
  transform: Transform,
  container: Size,
  image: Size,
): Transform {
  const scaledWidth = image.width * transform.scale
  const scaledHeight = image.height * transform.scale

  const tx =
    scaledWidth <= container.width
      ? (container.width - scaledWidth) / 2
      : Math.min(0, Math.max(container.width - scaledWidth, transform.tx))

  const ty =
    scaledHeight <= container.height
      ? (container.height - scaledHeight) / 2
      : Math.min(0, Math.max(container.height - scaledHeight, transform.ty))

  return { scale: transform.scale, tx, ty }
}

/** Transform that fits the image in the container and centres it. */
export function fitTransform(container: Size, image: Size): Transform {
  const scale = computeFitScale(container, image)
  return clampTranslation({ scale, tx: 0, ty: 0 }, container, image)
}

/**
 * Change scale while keeping the point under `anchor` stationary.
 *
 * This is what makes wheel zoom feel correct: the pixel under the cursor stays
 * under the cursor. Anchoring at the container centre instead gives the
 * behaviour expected from the zoom buttons.
 */
export function zoomAtPoint(
  transform: Transform,
  nextScale: number,
  anchor: { x: number; y: number },
  container: Size,
  image: Size,
): Transform {
  const scale = clampScale(nextScale)
  const ratio = scale / transform.scale

  return clampTranslation(
    {
      scale,
      tx: anchor.x - (anchor.x - transform.tx) * ratio,
      ty: anchor.y - (anchor.y - transform.ty) * ratio,
    },
    container,
    image,
  )
}

/** Zoom about the container's centre. */
export function zoomToScale(
  transform: Transform,
  nextScale: number,
  container: Size,
  image: Size,
): Transform {
  return zoomAtPoint(
    transform,
    nextScale,
    { x: container.width / 2, y: container.height / 2 },
    container,
    image,
  )
}

/**
 * Next preset above the current scale, or `null` at the top.
 *
 * A small epsilon avoids getting stuck when the current scale is a preset that
 * floating-point arithmetic has nudged imperceptibly.
 */
export function nextPresetUp(scale: number): number | null {
  return ZOOM_PRESETS.find((preset) => preset > scale + 1e-6) ?? null
}

export function nextPresetDown(scale: number): number | null {
  const lower = ZOOM_PRESETS.filter((preset) => preset < scale - 1e-6)
  return lower.length > 0 ? (lower[lower.length - 1] as number) : null
}

/**
 * Structural equality for transforms.
 *
 * Used to skip redundant state updates: returning a fresh object from a state
 * setter on every render is what turns a size-dependent effect into an
 * infinite render loop.
 */
export function transformsEqual(a: Transform, b: Transform): boolean {
  return a.scale === b.scale && a.tx === b.tx && a.ty === b.ty
}

/** True when the scaled image overflows the container on either axis. */
export function canPan(transform: Transform, container: Size, image: Size): boolean {
  // Before the container has been measured there is no viewport to pan within,
  // so panning is not possible regardless of the image size.
  if (container.width <= 0 || container.height <= 0) return false

  return (
    image.width * transform.scale > container.width + 0.5 ||
    image.height * transform.scale > container.height + 0.5
  )
}
