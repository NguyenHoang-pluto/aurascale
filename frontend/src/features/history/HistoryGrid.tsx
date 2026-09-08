import { Clock, History } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { EmptyState } from '@/components/feedback/EmptyState'
import { ProblemErrorPanel } from '@/components/feedback/ProblemErrorPanel'
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

/**
 * The filter values, paired with the key that names each one.
 *
 * The value is what the API is asked for and never changes with language; the
 * key is what the user reads.
 */
const FILTERS = [
  { value: 'any', key: 'filter.all' },
  { value: 'completed', key: 'filter.completed' },
  { value: 'failed', key: 'filter.failed' },
  { value: 'cancelled', key: 'filter.cancelled' },
] as const

type FilterValue = (typeof FILTERS)[number]['value']

export function HistoryGrid({ onView }: { onView: (job: JobRecord) => void }) {
  const { t } = useTranslation(['history', 'common'])
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
          label={t('history:filter.label')}
          value={filter}
          onChange={changeFilter}
          options={FILTERS.map((option) => ({
            value: option.value,
            label: t(`history:${option.key}`),
          }))}
          size="sm"
          className="w-auto"
        />
        {total > 0 && (
          <p className="text-xs text-muted-foreground">
            {t('history:showing', { first: range.first, last: range.last, total })}
          </p>
        )}
      </div>

      <p className="flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-xs text-muted-foreground">
        <Clock aria-hidden="true" className="size-3.5 shrink-0" />
        {t('history:retention')}
      </p>

      {history.isError && history.problem !== undefined && (
        <ProblemErrorPanel
          problem={history.problem}
          title={t('history:error.loadTitle')}
          onRetry={history.refetch}
        />
      )}

      {deletion.problem !== undefined && (
        <ProblemErrorPanel
          problem={deletion.problem}
          title={t('history:error.deleteTitle')}
        />
      )}

      {history.isPending && (
        <div
          aria-busy="true"
          aria-label={t('history:loading')}
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
          title={
            filter === 'any'
              ? t('history:empty.title')
              : t('history:empty.filteredTitle')
          }
          description={
            filter === 'any'
              ? t('history:empty.description')
              : t('history:empty.filteredDescription')
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
        <nav
          aria-label={t('history:pager.label')}
          className="flex items-center justify-center gap-2"
        >
          <Button
            variant="secondary"
            size="sm"
            onClick={() => { setPage((current) => Math.max(0, current - 1)) }}
            disabled={page === 0}
          >
            {t('common:actions.previous')}
          </Button>
          <span className="text-xs text-muted-foreground">
            {t('history:pager.position', { page: page + 1, pages })}
          </span>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => { setPage((current) => Math.min(pages - 1, current + 1)) }}
            disabled={page >= pages - 1}
          >
            {t('common:actions.next')}
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
  const { t } = useTranslation(['history', 'common'])

  return (
    <div
      role="alertdialog"
      aria-label={t('history:confirm.label')}
      className="flex flex-wrap items-center gap-3 rounded-lg border border-destructive/40 bg-destructive/8 p-4"
    >
      <p className="min-w-0 flex-1 text-sm">{t('history:confirm.question')}</p>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>
          {t('common:actions.keep')}
        </Button>
        <Button variant="destructive" size="sm" onClick={onConfirm} autoFocus>
          {t('common:actions.delete')}
        </Button>
      </div>
      <span className="sr-only">{job.jobId}</span>
    </div>
  )
}
