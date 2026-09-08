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

export interface CreateJobRequest {
  file: File
  model: string
  scale: number
  format: OutputFormat
  quality?: number
  preserveMetadata: boolean
  settings: EnhanceSettings
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

/** Human-readable stage names for the UI. Keep in step with `JobStage`. */
export const STAGE_LABELS: Record<JobStage, string> = {
  validating: 'Validating',
  loading_model: 'Loading model',
  preprocessing: 'Preparing image',
  running_inference: 'Enhancing',
  postprocessing: 'Post-processing',
  encoding: 'Encoding',
}
