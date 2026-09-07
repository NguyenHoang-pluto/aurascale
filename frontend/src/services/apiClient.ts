import { env } from '@/config/env'
import type { ErrorCode, ProblemDetail } from '@/types/api'

/**
 * The one place HTTP happens (§ 23). Every failure path — HTTP error, network
 * failure, abort, unparseable body — arrives at the caller as an `ApiError`
 * carrying a `ProblemDetail`, so UI code has a single shape to handle.
 */
export class ApiError extends Error {
  readonly problem: ProblemDetail

  constructor(problem: ProblemDetail) {
    super(problem.detail)
    this.name = 'ApiError'
    this.problem = problem
  }

  get code(): ErrorCode {
    return this.problem.code
  }

  get status(): number {
    return this.problem.status
  }

  /** True when the request never reached the backend. */
  get isConnectivityError(): boolean {
    return this.code === 'network_error' || this.code === 'backend_unavailable'
  }
}

function clientProblem(
  code: ErrorCode,
  title: string,
  detail: string,
  technical?: string,
): ProblemDetail {
  return {
    type: `https://pixelforge.ai/errors/${code}`,
    title,
    status: 0,
    code,
    detail,
    ...(technical !== undefined ? { technical } : {}),
  }
}

function isProblemDetail(value: unknown): value is ProblemDetail {
  if (typeof value !== 'object' || value === null) return false
  const candidate = value as Record<string, unknown>
  return typeof candidate['code'] === 'string' && typeof candidate['detail'] === 'string'
}

export interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: BodyInit | null
  /** Milliseconds before the request is aborted. Defaults to 30s; pass 0 to disable. */
  timeoutMs?: number
}

const DEFAULT_TIMEOUT_MS = 30_000

async function parseProblem(response: Response): Promise<ProblemDetail> {
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    return clientProblem(
      'malformed_response',
      'Unexpected response',
      'The server returned a response that could not be understood.',
      `HTTP ${response.status} ${response.statusText}`,
    )
  }

  if (isProblemDetail(payload)) return payload

  return clientProblem(
    'malformed_response',
    'Unexpected response',
    'The server returned an error in an unexpected format.',
    `HTTP ${response.status}: ${JSON.stringify(payload).slice(0, 500)}`,
  )
}

/**
 * Perform a request against the backend.
 *
 * @throws {ApiError} on any non-2xx response, network failure or timeout.
 */
export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { timeoutMs = DEFAULT_TIMEOUT_MS, signal, ...init } = options
  const url = `${env.apiBaseUrl}${path}`

  const controller = new AbortController()
  const timeoutId =
    timeoutMs > 0 ? setTimeout(() => { controller.abort('timeout') }, timeoutMs) : undefined

  // Honour a caller-supplied signal alongside our timeout.
  if (signal != null) {
    if (signal.aborted) controller.abort(signal.reason)
    else signal.addEventListener('abort', () => { controller.abort(signal.reason) }, { once: true })
  }

  let response: Response
  try {
    response = await fetch(url, { ...init, signal: controller.signal })
  } catch (cause) {
    if (controller.signal.aborted && controller.signal.reason === 'timeout') {
      throw new ApiError(
        clientProblem(
          'timeout',
          'Request timed out',
          'The server took too long to respond. It may still be starting up.',
          `${path} exceeded ${String(timeoutMs)}ms`,
        ),
      )
    }
    if (controller.signal.aborted) {
      throw new ApiError(
        clientProblem('request_cancelled', 'Request cancelled', 'The request was cancelled.'),
      )
    }
    throw new ApiError(
      clientProblem(
        'backend_unavailable',
        'Cannot reach the backend',
        'The PixelForge backend is not responding. Check that it is running.',
        cause instanceof Error ? cause.message : String(cause),
      ),
    )
  } finally {
    if (timeoutId !== undefined) clearTimeout(timeoutId)
  }

  if (!response.ok) throw new ApiError(await parseProblem(response))

  if (response.status === 204) return undefined as T

  try {
    return (await response.json()) as T
  } catch (cause) {
    throw new ApiError(
      clientProblem(
        'malformed_response',
        'Unexpected response',
        'The server returned a response that could not be understood.',
        cause instanceof Error ? cause.message : String(cause),
      ),
    )
  }
}

/** Normalise an unknown thrown value into an ApiError for display. */
export function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error
  return new ApiError(
    clientProblem(
      'internal_error',
      'Something went wrong',
      'An unexpected error occurred.',
      error instanceof Error ? error.message : String(error),
    ),
  )
}
