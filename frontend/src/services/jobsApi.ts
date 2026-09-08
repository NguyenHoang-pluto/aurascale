import { apiRequest } from './apiClient'
import { env } from '@/config/env'
import type { CreateJobRequest, JobCreated, JobPage, JobRecord, JobStatus } from '@/types/job'

/**
 * The job endpoints.
 *
 * Submission is `multipart/form-data`, so `Content-Type` is deliberately not
 * set — the browser has to add its own boundary parameter, and setting the
 * header by hand produces a body the server cannot parse.
 */
export function createJob(request: CreateJobRequest, signal?: AbortSignal): Promise<JobCreated> {
  const form = new FormData()
  form.append('image', request.file, request.file.name)
  form.append('model', request.model)
  form.append('scale', String(request.scale))
  form.append('format', request.format)
  form.append('preserveMetadata', String(request.preserveMetadata))

  if (request.quality !== undefined) form.append('quality', String(request.quality))
  form.append('settings', JSON.stringify(request.settings))

  return apiRequest<JobCreated>('/api/jobs', {
    method: 'POST',
    body: form,
    // An upload can be large and the server writes it to disk while streaming;
    // the default 30s is for JSON calls, not for this.
    timeoutMs: 120_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}

/** One job record. Also the polling fallback when SSE is unavailable. */
export function getJob(jobId: string, signal?: AbortSignal): Promise<JobRecord> {
  return apiRequest<JobRecord>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    timeoutMs: 10_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}

/**
 * A page of history, newest first.
 *
 * `status` is omitted rather than sent empty: the backend treats an absent
 * filter as "any state", and an empty string would be a validation error.
 */
export function listJobs(
  options: { limit?: number; offset?: number; status?: JobStatus } = {},
  signal?: AbortSignal,
): Promise<JobPage> {
  const query = new URLSearchParams()
  if (options.limit !== undefined) query.set('limit', String(options.limit))
  if (options.offset !== undefined) query.set('offset', String(options.offset))
  if (options.status !== undefined) query.set('status', options.status)

  const suffix = query.toString()
  return apiRequest<JobPage>(`/api/jobs${suffix === '' ? '' : `?${suffix}`}`, {
    timeoutMs: 15_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}

/** Cancel a running job, or delete a finished one. Both are `DELETE`. */
export function cancelJob(jobId: string, signal?: AbortSignal): Promise<void> {
  return apiRequest<void>(`/api/jobs/${encodeURIComponent(jobId)}`, {
    method: 'DELETE',
    timeoutMs: 10_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}

/**
 * Where the finished image lives.
 *
 * Returned as a URL rather than fetched into memory: a 200 MP PNG has no
 * business passing through JavaScript just to be saved, and the browser's own
 * download handles `Content-Disposition` and resumption for us.
 */
export function resultUrl(jobId: string): string {
  return `${env.apiBaseUrl}/api/jobs/${encodeURIComponent(jobId)}/result`
}

/**
 * A view-sized copy of the result.
 *
 * The comparison viewer's base layer: capped on its long edge by the server,
 * so a 200 MP result never reaches the DOM whole.
 */
export function previewUrl(jobId: string): string {
  return `${env.apiBaseUrl}/api/jobs/${encodeURIComponent(jobId)}/preview`
}

/** A full-resolution slice of the result, for inspecting detail above 100%. */
export function cropUrl(
  jobId: string,
  region: { x: number; y: number; w: number; h: number },
): string {
  const query = new URLSearchParams({
    x: String(region.x),
    y: String(region.y),
    w: String(region.w),
    h: String(region.h),
  })
  return `${previewUrl(jobId)}?${query.toString()}`
}

/**
 * The history grid's tile: 256 px on the long edge, built once and cached.
 *
 * A URL rather than a fetch, so the browser loads, caches and lazily decodes
 * it like any other image.
 */
export function thumbnailUrl(jobId: string): string {
  return `${env.apiBaseUrl}/api/jobs/${encodeURIComponent(jobId)}/thumbnail`
}

/** The Server-Sent Events endpoint for a job's progress. */
export function eventsUrl(jobId: string): string {
  return `${env.apiBaseUrl}/api/jobs/${encodeURIComponent(jobId)}/events`
}
