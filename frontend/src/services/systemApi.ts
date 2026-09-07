import { apiRequest } from './apiClient'
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

/** Registered models and whether their weights are on disk. */
export function getModels(signal?: AbortSignal): Promise<ModelInfo[]> {
  return apiRequest<ModelInfo[]>('/api/models', {
    timeoutMs: 10_000,
    ...(signal !== undefined ? { signal } : {}),
  })
}
