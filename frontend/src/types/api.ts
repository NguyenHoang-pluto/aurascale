/**
 * Transport-level API types.
 *
 * `ErrorCode` mirrors `app/core/exceptions.py::ErrorCode` on the backend. The
 * two are kept in sync by hand; the API test suite asserts the backend never
 * emits a code outside this set.
 */

export type ErrorCode =
  // input validation
  | 'unsupported_format'
  | 'corrupted_image'
  | 'file_too_large'
  | 'image_too_large'
  | 'image_too_small'
  | 'output_too_large'
  | 'invalid_parameters'
  // resources
  | 'out_of_memory'
  | 'gpu_unavailable'
  | 'storage_full'
  // model / inference
  | 'model_not_found'
  | 'model_load_failed'
  | 'model_download_failed'
  | 'inference_failed'
  // jobs
  | 'job_not_found'
  | 'job_not_completed'
  | 'job_cancelled'
  | 'queue_full'
  | 'timeout'
  // catch-all
  | 'internal_error'
  // client-side only: never sent by the backend
  | 'network_error'
  | 'backend_unavailable'
  | 'request_cancelled'
  | 'malformed_response'

/** RFC 9457 problem document, as returned by every failing endpoint. */
export interface ProblemDetail {
  type: string
  title: string
  status: number
  code: ErrorCode
  detail: string
  technical?: string
  context?: Record<string, unknown>
}

export interface HealthResponse {
  status: string
  version: string
  environment: string
  uptimeSeconds: number
}
