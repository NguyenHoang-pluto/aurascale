import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { toApiError } from '@/services/apiClient'
import { cancelJob, listJobs } from '@/services/jobsApi'
import type { ProblemDetail } from '@/types/api'
import type { JobPage, JobStatus } from '@/types/job'

/**
 * A page of history.
 *
 * Server state, so it belongs to TanStack Query (docs/architecture.md § 11).
 * Paging and filtering are passed in rather than held here: they are view
 * state owned by the page, and putting them in a shared store would make two
 * screens fight over one filter.
 *
 * The query key carries the page and filter, so moving between pages caches
 * each one rather than discarding the last.
 */

export const HISTORY_QUERY_KEY = 'history'

export const PAGE_SIZE = 20

export interface HistoryQuery {
  page: number
  status?: JobStatus | undefined
}

export interface HistoryResult {
  data: JobPage | undefined
  isPending: boolean
  isError: boolean
  problem: ProblemDetail | undefined
  refetch: () => void
}

export function historyKey(query: HistoryQuery) {
  return [HISTORY_QUERY_KEY, query.page, query.status ?? 'any'] as const
}

export function useHistory(query: HistoryQuery): HistoryResult {
  const result = useQuery({
    queryKey: historyKey(query),
    queryFn: ({ signal }) =>
      listJobs(
        {
          limit: PAGE_SIZE,
          offset: query.page * PAGE_SIZE,
          ...(query.status !== undefined ? { status: query.status } : {}),
        },
        signal,
      ),
    // A backend that is down is an ordinary development state here.
    retry: false,
    // History changes when a job finishes elsewhere in the app, so it is not
    // treated as fresh for long.
    staleTime: 5_000,
  })

  return {
    data: result.data,
    isPending: result.isPending,
    isError: result.isError,
    problem: result.error != null ? toApiError(result.error).problem : undefined,
    refetch: () => { void result.refetch() },
  }
}

export interface DeleteJobResult {
  remove: (jobId: string) => void
  isDeleting: boolean
  deletingId: string | null
  problem: ProblemDetail | undefined
}

/**
 * Removing a job from history.
 *
 * The same `DELETE /api/jobs/{id}` the workspace uses. Every history page is
 * invalidated afterwards rather than just the current one: deleting shifts
 * every later job up by one, so the cached pages behind this one are wrong.
 */
export function useDeleteJob(): DeleteJobResult {
  const queryClient = useQueryClient()

  const mutation = useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: [HISTORY_QUERY_KEY] })
    },
  })

  return {
    remove: (jobId: string) => { mutation.mutate(jobId) },
    isDeleting: mutation.isPending,
    deletingId: mutation.isPending ? (mutation.variables ?? null) : null,
    problem: mutation.error != null ? toApiError(mutation.error).problem : undefined,
  }
}
