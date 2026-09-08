import { ApiError, apiRequest, clientProblem } from './apiClient'
import type { HealthResponse } from '@/types/api'
import type { ModelInfo, SystemInfo } from '@/types/system'

/** Liveness. Kept on a short timeout because it drives a status indicator. */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiRequest<HealthResponse>('/api/health', {
    timeoutMs: 5_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}

/**
 * Hardware and runtime capabilities.
 *
 * Slower than health: the first call imports torch on the server, which can
 * take several seconds on a cold process.
 */
export function getSystemInfo(signal?: AbortSignal): Promise<SystemInfo> {
  return apiRequest<SystemInfo>('/api/system', {
    timeoutMs: 20_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}

/**
 * Check that a model entry carries the fields the UI will read.
 *
 * `apiRequest` casts the parsed body to the expected type without looking at
 * it, which is fine while both sides agree. They do not always agree: a dev
 * server left running from an older release answers on the same port and
 * returns models with no `supportedScales`, and the cast let that `undefined`
 * travel all the way into the enhancement store before anything noticed.
 *
 * Checked here, at the edge, so a mismatched backend produces the ordinary
 * "cannot load the model list" panel instead of a blank page.
 */
function invalidModel(value: unknown): string | null {
  if (typeof value !== 'object' || value === null) return 'not an object'
  const model = value as Record<string, unknown>

  for (const field of ['id', 'name', 'description', 'arch'] as const) {
    if (typeof model[field] !== 'string') return `${field} is not a string`
  }
  if (typeof model['scale'] !== 'number') return 'scale is not a number'
  if (typeof model['supportsDenoise'] !== 'boolean') return 'supportsDenoise is not a boolean'
  if (typeof model['downloaded'] !== 'boolean') return 'downloaded is not a boolean'

  // An empty list is a real state — a model that can produce no factor at all
  // is unusable but describable, and the UI disables every option for it.
  // A missing list is not a state; it is an older server.
  const scales = model['supportedScales']
  if (!Array.isArray(scales) || scales.some((scale) => typeof scale !== 'number')) {
    return 'supportedScales is not an array of numbers'
  }

  return null
}

/** Registered models and whether their weights are on disk. */
export async function getModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  const payload = await apiRequest<unknown>('/api/models', {
    timeoutMs: 10_000,
    ...(signal !== undefined ? { signal } : {}),
  })

  if (!Array.isArray(payload)) {
    throw new ApiError(
      clientProblem(
        'malformed_response',
        'Unexpected response',
        'The server did not return a list of models.',
        `GET /api/models returned ${typeof payload}`,
      ),
    )
  }

  for (const [index, model] of payload.entries()) {
    const reason = invalidModel(model)
    if (reason !== null) {
      throw new ApiError(
        clientProblem(
          'malformed_response',
          'Unexpected response',
          'The model list does not match what this build expects. The backend may be an older version.',
          `GET /api/models: item ${String(index)} — ${reason}`,
        ),
      )
    }
  }

  return payload as ModelInfo[]
}
