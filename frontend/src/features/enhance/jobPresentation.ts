import type { TFunction } from 'i18next'
import type { SegmentedOption } from '@/components/ui/segmented-control'
import type { ImageMetadata } from '@/types/image'
import type { JobStatus } from '@/types/job'
import { stageKey } from '@/types/job'
import type { LiveProgress } from './useJobProgress'

/**
 * The wording and availability rules behind the enhancement panels.
 *
 * Kept apart from the components so they can be tested as functions, and so
 * each component file exports only components.
 *
 * The ones that produce a sentence take `t` rather than returning a key and
 * leaving the caller to assemble it: a caption like "Enhancing · tile 1 of 2"
 * is one phrase whose word order differs by language, and splitting it across
 * the call site would fix English order into every locale.
 */

/** Every factor the product offers, in order. Availability is per model. */
export const ALL_SCALES = [2, 4, 8] as const

/**
 * The upscale choices, with anything this model cannot produce disabled.
 *
 * `supported` comes from the backend, which derives it from the same planner
 * that validates a job. Nothing here restates that rule.
 */
export function scaleOptions(
  t: TFunction,
  supported: readonly number[],
): SegmentedOption<string>[] {
  return ALL_SCALES.map((scale) => ({
    value: String(scale),
    // A factor is a number and an "x". The same in every language.
    label: `${scale}x`,
    disabled: !supported.includes(scale),
    ...(scale === 8 ? { hint: t('enhance:scale.twoPassHint') } : {}),
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

/** The `job:status.*` key for a status. The caller translates it. */
export function statusKey(status: JobStatus): string {
  return `status.${status}`
}

/**
 * The caption under the progress bar.
 *
 * During inference it names the tile, because the tile count is what the
 * percentage is made of; elsewhere it names the stage and stops there rather
 * than implying a granularity that does not exist.
 */
export function describeProgress(
  t: TFunction,
  live: LiveProgress,
  status: JobStatus,
): string {
  if (status === 'queued') return t('job:progress.queued')

  if (live.stage === null) return t(`job:${statusKey(status)}`)
  const stage = t(`job:${stageKey(live.stage)}`)

  if (live.tilesDone !== null && live.tilesTotal !== null) {
    return t('job:progress.tile', {
      stage,
      done: live.tilesDone,
      total: live.tilesTotal,
    })
  }
  return stage
}

/** Why "Enhance" is unavailable, in one sentence, or undefined when it is not. */
export function describeBlocker(
  t: TFunction,
  hasImage: boolean,
  selected: { downloaded: boolean; name: string } | undefined,
): string | undefined {
  if (!hasImage) return t('job:blocked.noImage')
  if (selected === undefined) return t('job:blocked.noModels')
  if (!selected.downloaded) return t('job:blocked.notDownloaded', { model: selected.name })
  return undefined
}
