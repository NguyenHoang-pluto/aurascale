import { Download, Eye, ImageOff, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { StatusIndicator } from '@/components/ui/status'
import { formatDuration } from '@/lib/format'
import { cn } from '@/lib/cn'
import { resultUrl, thumbnailUrl } from '@/services/jobsApi'
import type { JobRecord } from '@/types/job'
import {
  describeSizes,
  formatTimestamp,
  hasResult,
  isRunning,
  statusLabel,
  statusTone,
} from './historyPresentation'

/**
 * One entry in the history grid.
 *
 * Only a completed job with an output offers view and download; a failed or
 * cancelled one has no file, and offering the buttons anyway would promise
 * something that is not there.
 *
 * Delete is withheld while a job is still running. `DELETE` on a running job
 * *cancels* it (Phase 7), which is a different action with a different
 * meaning, and it belongs in the workspace where the job is being watched.
 */
export function HistoryCard({
  job,
  onView,
  onDelete,
  isDeleting = false,
}: {
  job: JobRecord
  onView: (job: JobRecord) => void
  onDelete: (job: JobRecord) => void
  isDeleting?: boolean
}) {
  const [thumbnailFailed, setThumbnailFailed] = useState(false)
  const showThumbnail = hasResult(job) && !thumbnailFailed

  return (
    <article
      aria-label={`Job from ${formatTimestamp(job.createdAt)}`}
      className={cn(
        'flex flex-col overflow-hidden rounded-lg border border-border bg-surface',
        isDeleting && 'opacity-50',
      )}
    >
      <div className="relative aspect-4/3 shrink-0 overflow-hidden bg-canvas">
        {showThumbnail ? (
          <img
            src={thumbnailUrl(job.jobId)}
            alt={`Result of the job from ${formatTimestamp(job.createdAt)}`}
            loading="lazy"
            decoding="async"
            onError={() => { setThumbnailFailed(true) }}
            className="size-full object-contain"
          />
        ) : (
          // No file, or one the server could not produce a tile for. Said
          // plainly rather than left as a broken image.
          <div className="flex size-full flex-col items-center justify-center gap-1.5 text-muted-foreground">
            <ImageOff aria-hidden="true" className="size-5" />
            <span className="text-xs">
              {hasResult(job) ? 'Preview unavailable' : 'No result'}
            </span>
          </div>
        )}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <StatusIndicator tone={statusTone(job.status)} label={statusLabel(job.status)} />
          <Badge tone="neutral">{job.scale}x</Badge>
        </div>

        <dl className="flex flex-col gap-1 text-xs">
          <Row label="Model">
            <span className="truncate" title={job.model}>
              {job.model}
            </span>
          </Row>
          <Row label="Size">
            <MetricBadge>{describeSizes(job)}</MetricBadge>
          </Row>
          {job.processingMs !== null && (
            <Row label="Took">
              <MetricBadge>{formatDuration(job.processingMs)}</MetricBadge>
            </Row>
          )}
          <Row label="Created">
            <span className="text-muted-foreground">{formatTimestamp(job.createdAt)}</span>
          </Row>
        </dl>

        {job.error !== null && (
          <p className="line-clamp-2 text-xs text-destructive" title={job.error.detail}>
            {job.error.detail}
          </p>
        )}

        <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-1">
          {hasResult(job) && (
            <>
              <Button variant="secondary" size="sm" onClick={() => { onView(job) }}>
                <Eye aria-hidden="true" />
                View
              </Button>
              <Button asChild variant="ghost" size="sm">
                <a href={resultUrl(job.jobId)} download>
                  <Download aria-hidden="true" />
                  Download
                </a>
              </Button>
            </>
          )}

          {!isRunning(job) && (
            <Button
              variant="ghost"
              size="sm"
              className="ml-auto text-muted-foreground hover:text-destructive"
              onClick={() => { onDelete(job) }}
              disabled={isDeleting}
              aria-label={`Delete the job from ${formatTimestamp(job.createdAt)}`}
            >
              <Trash2 aria-hidden="true" />
              {isDeleting ? 'Deleting…' : 'Delete'}
            </Button>
          )}
        </div>
      </div>
    </article>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="shrink-0 text-muted-foreground">{label}</dt>
      <dd className="min-w-0 truncate text-right">{children}</dd>
    </div>
  )
}
