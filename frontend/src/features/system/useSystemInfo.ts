import { useQuery } from '@tanstack/react-query'
import { toApiError } from '@/services/apiClient'
import { getModels, getSystemInfo } from '@/services/systemApi'
import type { ModelInfo, SystemInfo } from '@/types/system'

export const SYSTEM_QUERY_KEY = ['system', 'info'] as const
export const MODELS_QUERY_KEY = ['system', 'models'] as const

export interface SystemInfoResult {
  data: SystemInfo | undefined
  isPending: boolean
  isError: boolean
  /** Human-readable failure reason, in the same shape as any other API error. */
  reason: string | undefined
  refetch: () => void
}

/**
 * Hardware and runtime capabilities from `GET /api/system`.
 *
 * Refetched periodically because free VRAM changes as jobs run — it is the
 * number that decides whether the next job needs a smaller tile size. The rest
 * of the payload is effectively static, so the interval is unhurried.
 */
export function useSystemInfo(): SystemInfoResult {
  const query = useQuery({
    queryKey: SYSTEM_QUERY_KEY,
    queryFn: ({ signal }) => getSystemInfo(signal),
    refetchInterval: 30_000,
    // A backend that is down is a normal development state, not something to
    // retry hard against.
    retry: false,
    staleTime: 15_000,
  })

  return {
    data: query.data,
    isPending: query.isPending,
    isError: query.isError,
    reason: query.error != null ? toApiError(query.error).problem.detail : undefined,
    refetch: () => {
      void query.refetch()
    },
  }
}

export interface ModelsResult {
  data: ModelInfo[] | undefined
  isPending: boolean
  isError: boolean
  reason: string | undefined
}

/** Registered models. Static for a process's lifetime apart from downloads. */
export function useModels(): ModelsResult {
  const query = useQuery({
    queryKey: MODELS_QUERY_KEY,
    queryFn: ({ signal }) => getModels(signal),
    retry: false,
    staleTime: 60_000,
  })

  return {
    data: query.data,
    isPending: query.isPending,
    isError: query.isError,
    reason: query.error != null ? toApiError(query.error).problem.detail : undefined,
  }
}
