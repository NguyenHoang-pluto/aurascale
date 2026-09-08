import type { StatusTone } from '@/components/ui/status'
import type { JobRecord, JobStatus } from '@/types/job'

/**
 * The wording and availability rules for a history entry.
 *
 * Kept apart from the components so they can be tested as functions, and so
 * each component file exports only components.
 */

const STATUS_TONE: Record<JobStatus, StatusTone> = {
  queued: 'pending',
  processing: 'pending',
  completed: 'success',
  failed: 'danger',
  cancelled: 'warning',
}

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  processing: 'In progress',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function statusTone(status: JobStatus): StatusTone {
  return STATUS_TONE[status]
}

export function statusLabel(status: JobStatus): string {
  return STATUS_LABEL[status]
}

/** Whether a job is still moving, and so must not be deleted from here. */
export function isRunning(job: JobRecord): boolean {
  return job.status === 'queued' || job.status === 'processing'
}

/**
 * Whether this entry has a result to view or download.
 *
 * Both need the output to exist: a failed or cancelled job has none, and
 * offering the actions anyway would promise a file that is not there.
 */
export function hasResult(job: JobRecord): boolean {
  return job.status === 'completed' && job.output !== null
}

/**
 * A short, absolute timestamp.
 *
 * Absolute rather than "3 hours ago": with a 24-hour retention window the
 * actual time tells the user how long an entry has left, which a relative
 * phrase hides.
 */
export function formatTimestamp(iso: string, now: Date = new Date()): string {
  const at = new Date(iso)
  if (Number.isNaN(at.getTime())) return 'Unknown'

  const time = at.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  const sameDay =
    at.getFullYear() === now.getFullYear() &&
    at.getMonth() === now.getMonth() &&
    at.getDate() === now.getDate()

  if (sameDay) return `Today ${time}`
  return `${at.toLocaleDateString(undefined, { day: 'numeric', month: 'short' })} ${time}`
}

/** "1,280 × 720 → 5,120 × 2,880", or just the input when there is no output. */
export function describeSizes(job: JobRecord): string {
  const input = `${job.input.width.toLocaleString()} × ${job.input.height.toLocaleString()}`
  if (job.output === null) return input

  const output = `${job.output.width.toLocaleString()} × ${job.output.height.toLocaleString()}`
  return `${input} → ${output}`
}

/** How many pages the pager should offer for a given total. */
export function pageCount(total: number, pageSize: number): number {
  if (pageSize <= 0) return 1
  return Math.max(1, Math.ceil(total / pageSize))
}

/** The 1-based range this page covers, for "showing 1-20 of 42". */
export function pageRange(
  page: number,
  pageSize: number,
  total: number,
): { first: number; last: number } {
  if (total === 0) return { first: 0, last: 0 }

  const first = page * pageSize + 1
  return { first, last: Math.min(total, first + pageSize - 1) }
}
