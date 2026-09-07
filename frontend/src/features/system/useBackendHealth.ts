import { useQuery } from '@tanstack/react-query'
import { getHealth } from '@/services/systemApi'
import { toApiError } from '@/services/apiClient'
import type { HealthResponse } from '@/types/api'

export const HEALTH_QUERY_KEY = ['system', 'health'] as const

export type BackendConnectionState = 'checking' | 'online' | 'offline'

export interface BackendHealth {
  state: BackendConnectionState
  data: HealthResponse | undefined
  /** Human-readable reason the backend is unreachable, when it is. */
  reason: string | undefined
  refetch: () => void
}

/**
 * Poll `/api/health` so the top bar reflects real backend reachability.
 *
 * Reachability only. GPU and CUDA state come from `/api/system` via
 * `useSystemInfo`; the two are kept apart so the indicator can say the backend
 * is down without also making a claim about the device.
 */
export function useBackendHealth(): BackendHealth {
  const query = useQuery({
    queryKey: HEALTH_QUERY_KEY,
    queryFn: ({ signal }) => getHealth(signal),
    refetchInterval: 15_000,
    // A dead backend is a normal state during development, not an exception
    // worth retrying hard or logging noisily.
    retry: false,
    staleTime: 10_000,
  })

  const state: BackendConnectionState = query.isPending
    ? 'checking'
    : query.isError
      ? 'offline'
      : 'online'

  return {
    state,
    data: query.data,
    reason: query.error != null ? toApiError(query.error).problem.detail : undefined,
    refetch: () => {
      void query.refetch()
    },
  }
}
