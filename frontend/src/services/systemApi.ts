import { apiRequest } from './apiClient'
import type { HealthResponse } from '@/types/api'

/**
 * System and health endpoints.
 *
 * Only `/api/health` exists today. `getSystemInfo` (GPU, CUDA, VRAM) arrives
 * with the backend system endpoint in Phase 5; the status indicator reports
 * backend reachability until then rather than guessing at GPU state.
 */
export function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return apiRequest<HealthResponse>('/api/health', {
    // Health drives a status dot; a slow answer is as good as no answer.
    timeoutMs: 5_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}
