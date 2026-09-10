/**
 * Job types.
 *
 * These mirror the response schemas in `backend/app/schemas/job.py`. The
 * backend serialises camelCase, so nothing is renamed here.
 *
 * Storage paths are deliberately absent: the backend never sends them, and the
 * client has no business knowing where a file lives on the server.
 */

import type { ErrorCode } from './api'

export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled'

export type JobStage =
  | 'validating'
  | 'loading_model'
  | 'preprocessing'
  | 'running_inference'
  | 'postprocessing'
  | 'encoding'

export type OutputFormat = 'png' | 'jpeg' | 'webp'

/** Statuses a job can no longer move out of on its own. */
export const TERMINAL_STATUSES: readonly JobStatus[] = ['completed', 'failed', 'cancelled']

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_STATUSES.includes(status)
}

export interface ImageFacts {
  width: number
  height: number
  sizeBytes: number
  format: string
}

export interface JobError {
  code: ErrorCode
  detail: string
  technical?: string | null
}

export interface JobRecord {
  jobId: string
  status: JobStatus
  /** Null outside `processing`. */
  stage: JobStage | null
  progress: number
  model: string
  scale: number
  /** Null for a job submitted before modes existed, or without one. */
  mode: EnhancementMode | null
  /** Whether the size was asked for as a factor or as a destination. */
  outputType: OutputType
  /** The preset, when `outputType` is `target`. */
  target: TargetResolution | null
  /** Where it ran. Null until the job starts. */
  device: string | null
  input: ImageFacts
  output: ImageFacts | null
  /** Measured wall time, excluding queue wait. */
  processingMs: number | null
  error: JobError | null
  createdAt: string
  startedAt: string | null
  finishedAt: string | null
}

/** One page of history, as returned by `GET /api/jobs`. */
export interface JobPage {
  items: JobRecord[]
  /** Everything matching the filter, not just this page, so a pager can be sized. */
  total: number
  limit: number
  offset: number
}

export interface JobCreated {
  jobId: string
  status: JobStatus
  /** 0 means it is next, or already started. */
  queuePosition: number
  createdAt: string
}

/** The optional `settings` object of a submission. */
export interface EnhanceSettings {
  sharpenStrength?: number
  denoiseStrength?: number
  tileSize?: number | null
  tilePad?: number | null
}

/**
 * What the enhancement should prioritise.
 *
 * A named bundle of decisions the backend makes - which model, how much
 * denoising - so the UI never has to name a model to express an intent.
 */
export type EnhancementMode = 'standard' | 'creative'

/**
 * A requested output size, named by its long edge.
 *
 * Deliberately not an upscale factor. "4K" is a destination the output lands
 * on whatever the input was; "4x" is a multiplier. The two are different
 * questions and the UI keeps them apart.
 *
 * 16K makes that impossible to miss: it is eight times the 1920 the family is
 * built on, so a 1080p source reaches it with an 8x pass, not a 16x one.
 */
export type TargetResolution = '2k' | '4k' | '6k' | '8k' | '16k'

/**
 * How a job's output size was asked for.
 *
 * A job from before target resolutions existed carries no metadata and reads
 * as `scale`, which is what it was - that was the only way to ask.
 */
export type OutputType = 'scale' | 'target'

export const TARGET_LONG_EDGE: Record<TargetResolution, number> = {
  '2k': 1920,
  '4k': 3840,
  '6k': 5760,
  '8k': 7680,
  '16k': 15360,
}

/**
 * The model each mode runs when the user has not named one.
 *
 * A mirror of `STANDARD_MODEL` and `CREATIVE_MODEL` in
 * `backend/app/services/mode_planner.py`, and only a mirror: **the backend
 * remains authoritative**. It resolves the model itself and ignores anything
 * the client believes, so nothing here can change which weights run.
 *
 * What it is for is the opposite problem. The panel has to show a model before
 * the user picks one, and it used to show whichever model happened to be
 * adopted first - so Creative displayed `RealESRGAN_x4plus`, read
 * `supportsDenoise` off it, and disabled the noise-reduction slider, while the
 * job ran `realesr-general-x4v3` at denoise 0.25. Three statements on screen,
 * all wrong for the job that ran, and the one control Creative exists to expose
 * greyed out.
 *
 * A contract test on the backend reads this table out of the TypeScript source
 * and asserts the two agree, the same way it already does for
 * `TARGET_LONG_EDGE`, so a mode whose model changes on the server cannot drift
 * from what the browser shows.
 */
export const MODE_DEFAULT_MODEL: Record<EnhancementMode, string> = {
  standard: 'RealESRGAN_x4plus',
  creative: 'realesr-general-x4v3',
}

export interface CreateJobRequest {
  file: File
  /**
   * Omitted when the user has not chosen a model, which lets `mode` supply one.
   * The backend already accepts an absent model and falls back to the mode's
   * default and then to `default_model`.
   */
  model?: string
  /** Ignored when `target` is set - the two answer the same question. */
  scale: number
  format: OutputFormat
  quality?: number
  preserveMetadata: boolean
  settings: EnhanceSettings
  mode?: EnhancementMode
  /** When set, the backend plans the factor and resizes to this exactly. */
  target?: TargetResolution
}

/**
 * What the progress stream sends. Names match the SSE `event:` field.
 *
 * `tilesDone`/`tilesTotal` appear only during inference, because that is the
 * only stage with a real count behind it — the backend does not invent one for
 * the others, and neither does this type.
 */
export interface ProgressEventData {
  progress: number
  stage: JobStage
  tilesDone?: number
  tilesTotal?: number
}

export interface StageEventData {
  stage: JobStage
  progress: number
}

export interface CompletedEventData {
  jobId: string
  processingMs: number | null
}

export interface FailedEventData {
  code: ErrorCode
  detail: string
  technical?: string
}

export interface CancelledEventData {
  jobId: string
}

/**
 * The `job:stage.*` key naming a stage. The caller translates it.
 *
 * A key rather than a label, so the stage a job reports over SSE stays a
 * protocol value and only becomes words at the point it is shown.
 */
export function stageKey(stage: JobStage): string {
  return `stage.${stage}`
}
