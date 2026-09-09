import { MAX_OUTPUT_PIXELS } from '@/config/limits'
import { TARGET_LONG_EDGE } from '@/types/job'
import type { TargetResolution } from '@/types/job'

/**
 * Which output sizes a loaded image can actually reach.
 *
 * A mirror of the backend's planning, and only a mirror: **the backend remains
 * authoritative** and re-checks everything it receives. This exists so an
 * impossible choice is greyed out with a reason, rather than accepted and then
 * refused after the upload.
 *
 * Two questions are answered here, because they are the same question asked
 * two ways and neither should be able to drift from the other.
 *
 * **A target preset**, mirroring `plan_resolution` in
 * `backend/app/services/resolution_planner.py`, whose three rules are applied
 * in the same order:
 *
 *   1. an image already at or beyond the target has nothing to gain - upscaling
 *      cannot make it smaller, and a silent downscale would answer a question
 *      nobody asked;
 *   2. the factor needed must be one the selected model actually offers;
 *   3. the *neural* result must fit inside the pixel limit - it is the largest
 *      thing the job holds, and it exists before the final resize shrinks it.
 *
 * **A factor**, mirroring `EnhancementService.plan` and
 * `ImageService.assert_output_fits`:
 *
 *   1. the factor must be one the model publishes in `supportedScales` - which
 *      is not a rule restated here but the backend's own answer, derived on the
 *      server from the planner that validates a job;
 *   2. the result must fit inside the pixel limit. For a composed factor like
 *      8x or 16x the final pass is the largest, so checking the result covers
 *      every intermediate.
 *
 * A contract test on the backend reads the preset table out of this codebase
 * and asserts the two agree, so the long edges cannot drift apart silently.
 */

export type TargetUnavailableReason = 'alreadyLarger' | 'beyondMaxScale' | 'exceedsPixelLimit'

export interface TargetAvailability {
  target: TargetResolution
  available: boolean
  /** Why not. Undefined when it is available. */
  reason?: TargetUnavailableReason
  /** The factor the model would be asked for. Undefined when unavailable. */
  neuralScale?: number
  /** The size the user would receive. Undefined when unavailable. */
  width?: number
  height?: number
}

export const ALL_TARGETS: readonly TargetResolution[] = ['2k', '4k', '6k', '8k', '16k']

/** The exact output size for a preset, long edge fixed, aspect ratio kept. */
export function targetDimensions(
  width: number,
  height: number,
  target: TargetResolution,
): { width: number; height: number } {
  const longEdge = TARGET_LONG_EDGE[target]

  if (width >= height) {
    return { width: longEdge, height: Math.max(1, Math.round((height * longEdge) / width)) }
  }
  return { width: Math.max(1, Math.round((width * longEdge) / height)), height: longEdge }
}

/** Whether one preset is reachable, and what it would produce. */
export function planTarget(
  width: number,
  height: number,
  target: TargetResolution,
  supportedScales: readonly number[],
  maxOutputPixels: number = MAX_OUTPUT_PIXELS,
): TargetAvailability {
  const sourceLong = Math.max(width, height)
  const targetLong = TARGET_LONG_EDGE[target]

  if (width <= 0 || height <= 0) {
    return { target, available: false, reason: 'beyondMaxScale' }
  }

  if (sourceLong >= targetLong) {
    return { target, available: false, reason: 'alreadyLarger' }
  }

  const required = targetLong / sourceLong
  const usable = [...supportedScales].filter((scale) => scale >= required).sort((a, b) => a - b)
  const neuralScale = usable[0]

  if (neuralScale === undefined) {
    return { target, available: false, reason: 'beyondMaxScale' }
  }

  if (width * neuralScale * height * neuralScale > maxOutputPixels) {
    return { target, available: false, reason: 'exceedsPixelLimit', neuralScale }
  }

  const dimensions = targetDimensions(width, height, target)
  return { target, available: true, neuralScale, ...dimensions }
}

/** Every preset, in ascending order, with its availability. */
export function planTargets(
  width: number,
  height: number,
  supportedScales: readonly number[],
  maxOutputPixels: number = MAX_OUTPUT_PIXELS,
): TargetAvailability[] {
  return ALL_TARGETS.map((target) =>
    planTarget(width, height, target, supportedScales, maxOutputPixels),
  )
}

// ------------------------------------------------------------------- factors

export type ScaleUnavailableReason = 'unsupportedByModel' | 'exceedsPixelLimit'

export interface ScaleAvailability {
  scale: number
  available: boolean
  /** Why not. Undefined when it is available. */
  reason?: ScaleUnavailableReason
  /** The size the user would receive. Undefined without a loaded image. */
  width?: number
  height?: number
}

/** The source dimensions a factor is judged against. */
export interface SourceSize {
  width: number
  height: number
}

/**
 * Whether one factor can be run on this image, and what it would produce.
 *
 * `source` is optional because the model rule stands on its own: with no image
 * loaded there is nothing to measure against, and every factor the model
 * offers stays selectable. That is what the control did before the pixel limit
 * was considered here at all.
 */
export function planScale(
  scale: number,
  supportedScales: readonly number[],
  source?: SourceSize,
  maxOutputPixels: number = MAX_OUTPUT_PIXELS,
): ScaleAvailability {
  if (!supportedScales.includes(scale)) {
    return { scale, available: false, reason: 'unsupportedByModel' }
  }

  if (source === undefined || source.width <= 0 || source.height <= 0) {
    return { scale, available: true }
  }

  const width = source.width * scale
  const height = source.height * scale

  if (width * height > maxOutputPixels) {
    return { scale, available: false, reason: 'exceedsPixelLimit', width, height }
  }

  return { scale, available: true, width, height }
}
