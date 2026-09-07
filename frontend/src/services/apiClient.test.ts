import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, apiRequest, toApiError } from './apiClient'
import { jsonResponse } from '@/test/renderWithProviders'
import type { ProblemDetail } from '@/types/api'

const PROBLEM: ProblemDetail = {
  type: 'https://pixelforge.ai/errors/image_too_large',
  title: 'Image too large',
  status: 413,
  code: 'image_too_large',
  detail: 'Your image is 81 MP, which is larger than the 16 MP limit.',
  technical: 'input=9000x9000 limit=16000000',
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

function stubFetch(impl: () => Promise<Response>) {
  const spy = vi.fn(impl)
  vi.stubGlobal('fetch', spy)
  return spy
}

/**
 * A fetch that never resolves on its own but rejects when its signal aborts,
 * which is what the real implementation does. A stub that ignores the signal
 * would let an abort silently hang instead of surfacing as a rejection.
 */
function stubHangingFetch() {
  return stubFetch(
    (...args: unknown[]) =>
      new Promise<Response>((_resolve, reject) => {
        const init = args[1] as RequestInit | undefined
        const signal = init?.signal
        if (signal == null) return
        const onAbort = () => {
          reject(new DOMException('The operation was aborted.', 'AbortError'))
        }
        if (signal.aborted) onAbort()
        else signal.addEventListener('abort', onAbort, { once: true })
      }),
  )
}

describe('apiRequest', () => {
  it('returns the parsed body on success', async () => {
    stubFetch(() => Promise.resolve(jsonResponse({ status: 'ok' })))

    await expect(apiRequest<{ status: string }>('/api/health')).resolves.toEqual({
      status: 'ok',
    })
  })

  it('returns undefined for 204 responses without parsing a body', async () => {
    stubFetch(() => Promise.resolve(new Response(null, { status: 204 })))

    await expect(apiRequest('/api/jobs/abc')).resolves.toBeUndefined()
  })

  it('throws an ApiError carrying the problem document', async () => {
    stubFetch(() =>
      Promise.resolve(jsonResponse(PROBLEM, 413, 'application/problem+json')),
    )

    const error = await apiRequest('/api/jobs').catch((e: unknown) => e)

    expect(error).toBeInstanceOf(ApiError)
    const apiError = error as ApiError
    expect(apiError.code).toBe('image_too_large')
    expect(apiError.status).toBe(413)
    expect(apiError.message).toBe(PROBLEM.detail)
    expect(apiError.problem.technical).toBe('input=9000x9000 limit=16000000')
  })

  it('reports an unreachable backend rather than surfacing the fetch failure', async () => {
    stubFetch(() => Promise.reject(new TypeError('Failed to fetch')))

    const error = (await apiRequest('/api/health').catch((e: unknown) => e)) as ApiError

    expect(error.code).toBe('backend_unavailable')
    expect(error.isConnectivityError).toBe(true)
    expect(error.problem.detail).toContain('not responding')
    // The underlying cause stays available for the technical-details panel.
    expect(error.problem.technical).toBe('Failed to fetch')
  })

  it('converts a non-problem error body into a malformed_response error', async () => {
    stubFetch(() => Promise.resolve(jsonResponse({ oops: true }, 500)))

    const error = (await apiRequest('/api/health').catch((e: unknown) => e)) as ApiError

    expect(error.code).toBe('malformed_response')
    expect(error.problem.technical).toContain('oops')
  })

  it('converts an unparseable error body into a malformed_response error', async () => {
    stubFetch(() =>
      Promise.resolve(new Response('<html>502</html>', { status: 502 })),
    )

    const error = (await apiRequest('/api/health').catch((e: unknown) => e)) as ApiError

    expect(error.code).toBe('malformed_response')
    expect(error.problem.technical).toContain('502')
  })

  it('aborts and reports a timeout when the server does not respond', async () => {
    stubHangingFetch()

    const error = (await apiRequest('/api/health', { timeoutMs: 20 }).catch(
      (e: unknown) => e,
    )) as ApiError

    expect(error.code).toBe('timeout')
    expect(error.problem.detail).toContain('too long')
  })

  it('propagates an abort from a caller-supplied signal', async () => {
    stubHangingFetch()
    const controller = new AbortController()
    const promise = apiRequest('/api/health', { signal: controller.signal, timeoutMs: 0 })
    controller.abort()

    const error = (await promise.catch((e: unknown) => e)) as ApiError

    expect(error.code).toBe('request_cancelled')
  })
})

describe('toApiError', () => {
  it('passes ApiError through unchanged', () => {
    const original = new ApiError(PROBLEM)

    expect(toApiError(original)).toBe(original)
  })

  it('wraps an unknown throwable', () => {
    const wrapped = toApiError(new Error('boom'))

    expect(wrapped.code).toBe('internal_error')
    expect(wrapped.problem.technical).toBe('boom')
  })
})
