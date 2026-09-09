import { Download, Eye, ImageOff, Trash2 } from 'lucide-react'
import { useState } from 'react'
import type { TFunction } from 'i18next'
import { useTranslation } from 'react-i18next'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { StatusIndicator } from '@/components/ui/status'
import { useErrorMessage } from '@/i18n/errorMessages'
import { formatDuration } from '@/lib/format'
import { cn } from '@/lib/cn'
import { resultUrl, thumbnailUrl } from '@/services/jobsApi'
import type { JobRecord } from '@/types/job'
import {
  hasResult,
  isRunning,
  modeKey,
  outputRequest,
  sizeParts,
  statusKey,
  statusTone,
  timestampParts,
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
  const { t, i18n } = useTranslation(['history', 'job', 'common'])
  const translateError = useErrorMessage()
  const [thumbnailFailed, setThumbnailFailed] = useState(false)

  const locale = i18n.language
  const showThumbnail = hasResult(job) && !thumbnailFailed
  const timestamp = formatCreated(t, job.createdAt, locale)
  const sizes = sizeParts(job, locale)
  const output = outputRequest(job)
  const mode = modeKey(job)
  const errorText = job.error === null ? null : translateError(job.error).detail

  return (
    <article
      aria-label={t('history:card.label', { timestamp })}
      className={cn(
        'flex flex-col overflow-hidden rounded-lg border border-border bg-surface',
        isDeleting && 'opacity-50',
      )}
    >
      <div className="relative aspect-4/3 shrink-0 overflow-hidden bg-canvas">
        {showThumbnail ? (
          <img
            src={thumbnailUrl(job.jobId)}
            alt={t('history:card.thumbnailAlt', { timestamp })}
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
              {hasResult(job)
                ? t('history:card.previewUnavailable')
                : t('history:card.noResult')}
            </span>
          </div>
        )}
      </div>

      <div className="flex min-w-0 flex-1 flex-col gap-2 p-3">
        <div className="flex items-center justify-between gap-2">
          <StatusIndicator
            tone={statusTone(job.status)}
            label={t(`job:${statusKey(job.status)}`)}
          />
          {/* "4x" or "4K" - a factor or a preset name. Neither is translated. */}
          <Badge tone="neutral">{output.value}</Badge>
        </div>

        <dl className="flex flex-col gap-1 text-xs">
          {mode !== null ? (
            // A mode is what the user chose; the model id is an implementation
            // detail they did not, so it is not shown when a mode exists.
            <Row label={t('history:card.mode')}>
              <span>{t(`history:${mode}`)}</span>
            </Row>
          ) : (
            // A job from before modes existed has no other provenance, so its
            // model is still the most useful thing to name.
            <Row label={t('history:card.model')}>
              <span className="truncate" title={job.model}>
                {job.model}
              </span>
            </Row>
          )}
          <Row label={t('history:card.output')}>
            <span>
              {t(`history:output.${output.key}`)} · {output.value}
            </span>
          </Row>
          <Row label={t('history:card.size')}>
            <MetricBadge>
              {sizes.output === null
                ? sizes.input
                : t('history:sizes.change', { input: sizes.input, output: sizes.output })}
            </MetricBadge>
          </Row>
          {job.processingMs !== null && (
            <Row label={t('history:card.took')}>
              <MetricBadge>{formatDuration(job.processingMs, locale)}</MetricBadge>
            </Row>
          )}
          <Row label={t('history:card.created')}>
            <span className="text-muted-foreground">{timestamp}</span>
          </Row>
        </dl>

        {errorText !== null && (
          <p className="line-clamp-2 text-xs text-destructive" title={errorText}>
            {errorText}
          </p>
        )}

        <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-1">
          {hasResult(job) && (
            <>
              <Button variant="secondary" size="sm" onClick={() => { onView(job) }}>
                <Eye aria-hidden="true" />
                {t('common:actions.view')}
              </Button>
              <Button asChild variant="ghost" size="sm">
                <a href={resultUrl(job.jobId)} download>
                  <Download aria-hidden="true" />
                  {t('common:actions.download')}
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
              aria-label={t('history:card.deleteLabel', { timestamp })}
            >
              <Trash2 aria-hidden="true" />
              {isDeleting ? t('history:card.deleting') : t('common:actions.delete')}
            </Button>
          )}
        </div>
      </div>
    </article>
  )
}

/**
 * A timestamp assembled from its parts, in the order this language uses.
 *
 * The parts are joined by a translation key rather than here, because "Today
 * 10:32" and "8 thg 9 10:32" do not put the pieces in the same order.
 */
function formatCreated(t: TFunction, iso: string, locale: string): string {
  const parts = timestampParts(iso, new Date(), locale)

  if (parts.key === 'unknown') return t('common:state.unknown')
  return t(`history:time.${parts.key}`, { time: parts.time, date: parts.date })
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="shrink-0 text-muted-foreground">{label}</dt>
      <dd className="min-w-0 truncate text-right">{children}</dd>
    </div>
  )
}
