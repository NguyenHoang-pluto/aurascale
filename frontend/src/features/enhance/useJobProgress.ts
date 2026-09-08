import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { eventsUrl, getJob } from '@/services/jobsApi'
import { toApiError } from '@/services/apiClient'
import type { ProblemDetail } from '@/types/api'
import type { JobRecord, JobStage, ProgressEventData, StageEventData } from '@/types/job'
import { isTerminal } from '@/types/job'

/**
 * Follow a job to its terminal state.
 *
 * Two channels, deliberately:
 *
 *  * **Server-Sent Events** carry every measurement the backend makes, as it
 *    makes it. This is the live one.
 *  * **`GET /api/jobs/{id}`** is polled as the fallback, at a slow interval
 *    while the stream is healthy and a brisk one when it is not, because SSE
 *    is the thing most likely to be blocked by a proxy.
 *
 * Nothing here invents a value. If neither channel has reported since the last
 * measurement, the bar stays where it is — a percentage that moves on a timer
 * would be a claim about work that has not happened.
 */

export const JOB_QUERY_KEY = (jobId: string) => ['job', jobId] as const

/** Slow, because the stream is doing the real work. */
const BACKGROUND_POLL_MS = 5_000
/** Brisk, because it is now the only channel. */
const FALLBACK_POLL_MS = 1_000

export interface LiveProgress {
  progress: number
  stage: JobStage | null
  tilesDone: number | null
  tilesTotal: number | null
}

export interface JobProgressResult {
  job: JobRecord | undefined
  /** The freshest measurement, from whichever channel reported last. */
  live: LiveProgress
  isStreaming: boolean
  problem: ProblemDetail | undefined
}

/** Whether this browser can open an event stream at all. */
export function supportsEventSource(): boolean {
  return typeof globalThis.EventSource === 'function'
}

export function useJobProgress(jobId: string | null): JobProgressResult {
  const queryClient = useQueryClient()
  // Tagged with the job it belongs to: when the active job changes, the old
  // snapshot is ignored during render rather than cleared by an effect, which
  // would show the previous job's numbers for one frame.
  const [streamed, setStreamed] = useState<(LiveProgress & { jobId: string }) | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const finishedRef = useRef(false)

  const query = useQuery({
    queryKey: jobId === null ? ['job', 'none'] : JOB_QUERY_KEY(jobId),
    queryFn: ({ signal }) => getJob(jobId as string, signal),
    enabled: jobId !== null,
    retry: false,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      if (status !== undefined && isTerminal(status)) return false
      return isStreaming ? BACKGROUND_POLL_MS : FALLBACK_POLL_MS
    },
  })

  useEffect(() => {
    if (jobId === null || !supportsEventSource()) return
    finishedRef.current = false

    const source = new EventSource(eventsUrl(jobId))

    const onProgress = (event: MessageEvent<string>) => {
      const data = parse<ProgressEventData>(event.data)
      if (data === null) return
      setStreamed({
        jobId,
        progress: data.progress,
        stage: data.stage,
        tilesDone: data.tilesDone ?? null,
        tilesTotal: data.tilesTotal ?? null,
      })
    }

    const onStage = (event: MessageEvent<string>) => {
      const data = parse<StageEventData>(event.data)
      if (data === null) return
      setStreamed({
        jobId,
        progress: data.progress,
        stage: data.stage,
        tilesDone: null,
        tilesTotal: null,
      })
    }

    const finish = () => {
      finishedRef.current = true
      // The stream is the fast path; the record is the authority on the
      // outcome, so the terminal event triggers one refetch rather than being
      // trusted to describe the result itself.
      void queryClient.invalidateQueries({ queryKey: JOB_QUERY_KEY(jobId) })
      source.close()
      setIsStreaming(false)
    }

    source.addEventListener('open', () => { setIsStreaming(true) })
    source.addEventListener('progress', onProgress as EventListener)
    source.addEventListener('stage', onStage as EventListener)
    source.addEventListener('completed', finish)
    source.addEventListener('failed', finish)
    source.addEventListener('cancelled', finish)
    source.addEventListener('error', () => {
      // EventSource reconnects on its own; the only thing to do here is stop
      // claiming the stream is healthy so polling speeds up.
      setIsStreaming(false)
      if (finishedRef.current) source.close()
    })

    return () => {
      source.close()
      setIsStreaming(false)
    }
  }, [jobId, queryClient])

  const job = query.data
  const current = streamed !== null && streamed.jobId === jobId ? streamed : null

  return {
    job,
    live: mergeProgress(job, current),
    isStreaming,
    problem: query.error != null ? toApiError(query.error).problem : undefined,
  }
}

/**
 * The freshest of the two channels.
 *
 * The record wins once the job is terminal — the stream's last progress event
 * is by then stale, and showing 90 % beside "Completed" would be nonsense.
 * Otherwise whichever reports the higher percentage is the more recent, since
 * progress only ever moves forward.
 */
export function mergeProgress(
  job: JobRecord | undefined,
  streamed: LiveProgress | null,
): LiveProgress {
  if (job !== undefined && isTerminal(job.status)) {
    return {
      progress: job.status === 'completed' ? 100 : job.progress,
      stage: null,
      tilesDone: null,
      tilesTotal: null,
    }
  }

  const fromRecord: LiveProgress = {
    progress: job?.progress ?? 0,
    stage: job?.stage ?? null,
    tilesDone: null,
    tilesTotal: null,
  }

  if (streamed === null) return fromRecord
  return streamed.progress >= fromRecord.progress ? streamed : fromRecord
}

function parse<T>(payload: string): T | null {
  try {
    return JSON.parse(payload) as T
  } catch {
    return null
  }
}
