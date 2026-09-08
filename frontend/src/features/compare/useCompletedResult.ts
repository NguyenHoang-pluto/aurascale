import { useEnhancementStore } from '@/stores/useEnhancementStore'
import { useComparisonStore, type BeforeSnapshot } from '@/stores/useComparisonStore'
import { useJobProgress } from '@/features/enhance/useJobProgress'

/**
 * Whether there is a finished result to compare, and what it consists of.
 *
 * Null unless a job genuinely completed *and* produced an output *and* we still
 * hold the snapshot of what went into it. Failed and cancelled jobs never
 * qualify: they have no result, and showing an image against itself would
 * claim work that did not happen.
 */
export interface CompletedResult {
  jobId: string
  before: BeforeSnapshot
  output: { width: number; height: number }
}

export function useCompletedResult(): CompletedResult | null {
  const activeJobId = useEnhancementStore((s) => s.activeJobId)
  const before = useComparisonStore((s) => s.before)
  const { job } = useJobProgress(activeJobId)

  if (activeJobId === null || before === null) return null
  if (job === undefined || job.status !== 'completed' || job.output === null) return null

  return {
    jobId: activeJobId,
    before,
    output: { width: job.output.width, height: job.output.height },
  }
}
