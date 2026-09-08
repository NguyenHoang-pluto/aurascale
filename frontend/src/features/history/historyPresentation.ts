import type { StatusTone } from '@/components/ui/status'
import { formatDimensions } from '@/lib/format'
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

export function statusTone(status: JobStatus): StatusTone {
  return STATUS_TONE[status]
}

/** The `job:status.*` key for a status. The caller translates it. */
export function statusKey(status: JobStatus): string {
  return `status.${status}`
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
export interface TimestampParts {
  /** Which `history:time.*` key to use. */
  key: 'today' | 'dated' | 'unknown'
  time: string
  date: string
}

/**
 * The pieces of a timestamp, for the caller to interpolate.
 *
 * Returns parts rather than a sentence: "Today 10:32" and "8 Sep 10:32" have
 * different word orders in different languages, and concatenating here would
 * bake English order into every locale.
 */
export function timestampParts(
  iso: string,
  now: Date = new Date(),
  locale?: string,
): TimestampParts {
  const at = new Date(iso)
  if (Number.isNaN(at.getTime())) return { key: 'unknown', time: '', date: '' }

  const time = at.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })
  const sameDay =
    at.getFullYear() === now.getFullYear() &&
    at.getMonth() === now.getMonth() &&
    at.getDate() === now.getDate()

  return {
    key: sameDay ? 'today' : 'dated',
    time,
    date: at.toLocaleDateString(locale, { day: 'numeric', month: 'short' }),
  }
}

/** The input and output dimensions, for the caller to join with a key. */
export function sizeParts(job: JobRecord, locale?: string): { input: string; output: string | null } {
  return {
    input: formatDimensions(job.input.width, job.input.height, locale),
    output:
      job.output === null
        ? null
        : formatDimensions(job.output.width, job.output.height, locale),
  }
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
