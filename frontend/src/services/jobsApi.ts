import { apiRequest } from './apiClient'
import { env } from '@/config/env'
import type { CreateJobRequest, JobCreated, JobRecord } from '@/types/job'

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

/** The Server-Sent Events endpoint for a job's progress. */
export function eventsUrl(jobId: string): string {
  return `${env.apiBaseUrl}/api/jobs/${encodeURIComponent(jobId)}/events`
}
