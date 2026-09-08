import type { SegmentedOption } from '@/components/ui/segmented-control'
import type { ImageMetadata } from '@/types/image'
import type { JobStatus } from '@/types/job'
import { STAGE_LABELS } from '@/types/job'
import type { LiveProgress } from './useJobProgress'

/**
 * The wording and availability rules behind the enhancement panels.
 *
 * Kept apart from the components so they can be tested as functions, and so
 * each component file exports only components.
 */

/** Every factor the product offers, in order. Availability is per model. */
export const ALL_SCALES = [2, 4, 8] as const

/**
 * The upscale choices, with anything this model cannot produce disabled.
 *
 * `supported` comes from the backend, which derives it from the same planner
 * that validates a job. Nothing here restates that rule.
 */
export function scaleOptions(supported: readonly number[]): SegmentedOption<string>[] {
  return ALL_SCALES.map((scale) => ({
    value: String(scale),
    label: `${scale}x`,
    disabled: !supported.includes(scale),
    ...(scale === 8 ? { hint: 'two-pass' } : {}),
  }))
}

/** The result's dimensions: exactly the input times the factor. */
export function projectedSize(
  metadata: ImageMetadata | undefined,
  scale: number,
): { width: number; height: number } | undefined {
  if (metadata === undefined) return undefined
  return { width: metadata.width * scale, height: metadata.height * scale }
}

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  processing: 'In progress',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function statusLabel(status: JobStatus): string {
  return STATUS_LABEL[status]
}

/**
 * The caption under the progress bar.
 *
 * During inference it names the tile, because the tile count is what the
 * percentage is made of; elsewhere it names the stage and stops there rather
 * than implying a granularity that does not exist.
 */
export function describeProgress(live: LiveProgress, status: JobStatus): string {
  if (status === 'queued') return 'Waiting for a free worker'

  const stage = live.stage === null ? null : STAGE_LABELS[live.stage]
  if (stage === null) return STATUS_LABEL[status]

  if (live.tilesDone !== null && live.tilesTotal !== null) {
    return `${stage} · tile ${String(live.tilesDone)} of ${String(live.tilesTotal)}`
  }
  return stage
}

/** Why "Enhance" is unavailable, in one sentence, or undefined when it is not. */
export function describeBlocker(
  hasImage: boolean,
  selected: { downloaded: boolean; name: string } | undefined,
): string | undefined {
  if (!hasImage) return 'Load an image to enhance.'
  if (selected === undefined) return 'Waiting for the model list.'
  if (!selected.downloaded) return `${selected.name} is not downloaded yet.`
  return undefined
}
