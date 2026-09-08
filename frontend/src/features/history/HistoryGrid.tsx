import { Clock, History } from 'lucide-react'
import { useState } from 'react'
import { EmptyState } from '@/components/feedback/EmptyState'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { Button } from '@/components/ui/button'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Skeleton } from '@/components/ui/skeleton'
import type { JobRecord, JobStatus } from '@/types/job'
import { HistoryCard } from './HistoryCard'
import { pageCount, pageRange } from './historyPresentation'
import { PAGE_SIZE, useDeleteJob, useHistory } from './useHistory'

/**
 * The history grid, with its filter and pager.
 *
 * Paging and the status filter are local state: they are how *this* screen is
 * being looked at, and hoisting them into a store would make another screen a
 * stakeholder in a decision that is none of its business.
 */

const FILTERS = [
  { value: 'any', label: 'All' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'cancelled', label: 'Cancelled' },
] as const

type FilterValue = (typeof FILTERS)[number]['value']

export function HistoryGrid({ onView }: { onView: (job: JobRecord) => void }) {
  const [page, setPage] = useState(0)
  const [filter, setFilter] = useState<FilterValue>('any')
  const [pendingDelete, setPendingDelete] = useState<JobRecord | null>(null)

  const status: JobStatus | undefined = filter === 'any' ? undefined : filter
  const history = useHistory({ page, status })
  const deletion = useDeleteJob()

  const total = history.data?.total ?? 0
  const pages = pageCount(total, PAGE_SIZE)
  const range = pageRange(page, PAGE_SIZE, total)

  const changeFilter = (value: FilterValue) => {
    setFilter(value)
    // A filter change makes the current offset meaningless.
    setPage(0)
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <SegmentedControl
          name="history-filter"
          label="Filter by status"
          value={filter}
          onChange={changeFilter}
          options={FILTERS.map((option) => ({ ...option }))}
          size="sm"
          className="w-auto"
        />
        {total > 0 && (
          <p className="text-xs text-muted-foreground">
            Showing {range.first}–{range.last} of {total}
          </p>
        )}
      </div>

      <p className="flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-xs text-muted-foreground">
        <Clock aria-hidden="true" className="size-3.5 shrink-0" />
        Jobs and their images are removed 24 hours after they finish. This is a
        record of recent work, not an archive — download anything you want to keep.
      </p>

      {history.isError && history.problem !== undefined && (
        <ErrorPanel
          title="Could not load history"
          detail={history.problem.detail}
          code={history.problem.code}
          {...(history.problem.technical !== undefined
            ? { technical: history.problem.technical }
            : {})}
          onRetry={history.refetch}
        />
      )}

      {deletion.problem !== undefined && (
        <ErrorPanel
          title="Could not delete that job"
          detail={deletion.problem.detail}
          code={deletion.problem.code}
        />
      )}

      {history.isPending && (
        <div
          aria-busy="true"
          aria-label="Loading history"
          className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3"
        >
          {Array.from({ length: 6 }, (_, index) => (
            <Skeleton key={index} className="h-72 w-full rounded-lg" />
          ))}
        </div>
      )}

      {!history.isPending && !history.isError && total === 0 && (
        <EmptyState
          icon={History}
          title={filter === 'any' ? 'Nothing here yet' : 'Nothing matches that filter'}
          description={
            filter === 'any'
              ? 'Enhance an image and it will appear here, alongside the model and settings it used.'
              : 'Try a different status.'
          }
        />
      )}

      {history.data !== undefined && history.data.items.length > 0 && (
        <ul className="grid list-none gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {history.data.items.map((job) => (
            <li key={job.jobId}>
              <HistoryCard
                job={job}
                onView={onView}
                onDelete={setPendingDelete}
                isDeleting={deletion.deletingId === job.jobId}
              />
            </li>
          ))}
        </ul>
      )}

      {pages > 1 && (
        <nav aria-label="History pages" className="flex items-center justify-center gap-2">
          <Button
            variant="secondary"
            size="sm"
            onClick={() => { setPage((current) => Math.max(0, current - 1)) }}
            disabled={page === 0}
          >
            Previous
          </Button>
          <span className="text-xs text-muted-foreground">
            Page {page + 1} of {pages}
          </span>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => { setPage((current) => Math.min(pages - 1, current + 1)) }}
            disabled={page >= pages - 1}
          >
            Next
          </Button>
        </nav>
      )}

      {pendingDelete !== null && (
        <ConfirmDelete
          job={pendingDelete}
          onCancel={() => { setPendingDelete(null) }}
          onConfirm={() => {
            deletion.remove(pendingDelete.jobId)
            setPendingDelete(null)
          }}
        />
      )}
    </div>
  )
}

/**
 * Confirmation before a deletion that cannot be undone.
 *
 * Inline rather than a modal: the app has no dialog primitive, and inventing
 * one here would be a second visual language for a two-button question.
 */
function ConfirmDelete({
  job,
  onCancel,
  onConfirm,
}: {
  job: JobRecord
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <div
      role="alertdialog"
      aria-label="Confirm deletion"
      className="flex flex-wrap items-center gap-3 rounded-lg border border-destructive/40 bg-destructive/8 p-4"
    >
      <p className="min-w-0 flex-1 text-sm">
        Delete this job and its image? The file is removed from the server and cannot
        be recovered.
      </p>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          Keep
        </Button>
        <Button variant="destructive" size="sm" onClick={onConfirm} autoFocus>
          Delete
        </Button>
      </div>
      <span className="sr-only">{job.jobId}</span>
    </div>
  )
}
