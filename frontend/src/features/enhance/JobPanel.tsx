import { Ban, Download, RotateCcw, Sparkles } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { ProblemErrorPanel } from '@/components/feedback/ProblemErrorPanel'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { StatusIndicator, type StatusTone } from '@/components/ui/status'
import { formatBytes, formatDimensions, formatDuration } from '@/lib/format'
import { resultUrl } from '@/services/jobsApi'
import type { ProblemDetail } from '@/types/api'
import type { JobRecord, JobStatus } from '@/types/job'
import { describeProgress, statusKey } from './jobPresentation'
import type { LiveProgress } from './useJobProgress'

/**
 * Submit, watch, and collect the result.
 *
 * Every number shown here is one the backend measured. While a job runs, the
 * bar is the percentage the server last reported and the caption names the
 * stage it is actually in; during inference it also shows the tile count,
 * because that is what the percentage is made of. Nothing advances on a timer.
 */

const STATUS_TONE: Record<JobStatus, StatusTone> = {
  queued: 'pending',
  processing: 'pending',
  completed: 'success',
  failed: 'danger',
  cancelled: 'warning',
}

export function JobPanel({
  job,
  live,
  canSubmit,
  disabledReason,
  isSubmitting,
  isCancelling,
  problem,
  onSubmit,
  onCancel,
  onReset,
}: {
  job: JobRecord | undefined
  live: LiveProgress
  canSubmit: boolean
  /** Why the button is disabled, said plainly rather than left to guesswork. */
  disabledReason?: string | undefined
  isSubmitting: boolean
  isCancelling: boolean
  problem: ProblemDetail | undefined
  onSubmit: () => void
  onCancel: () => void
  onReset: () => void
}) {
  const { t, i18n } = useTranslation(['job', 'errors', 'common'])
  const locale = i18n.language
  const isRunning =
    job !== undefined && (job.status === 'queued' || job.status === 'processing')
  const isFinished = job !== undefined && !isRunning

  return (
    <div className="flex flex-col gap-4">
      {job === undefined && (
        <>
          <Button
            className="w-full"
            onClick={onSubmit}
            disabled={!canSubmit || isSubmitting}
            aria-busy={isSubmitting}
          >
            <Sparkles aria-hidden="true" />
            {isSubmitting ? t('job:actions.submitting') : t('job:actions.submit')}
          </Button>
          {!canSubmit && disabledReason !== undefined && (
            <p className="text-xs text-muted-foreground">{disabledReason}</p>
          )}
        </>
      )}

      {job !== undefined && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <StatusIndicator
              live={isRunning}
              tone={STATUS_TONE[job.status]}
              label={t(`job:${statusKey(job.status)}`)}
            />
            {job.device !== null && <Badge tone="neutral">{job.device.toUpperCase()}</Badge>}
          </div>

          {isRunning && (
            <>
              <Progress value={live.progress} label={t('job:progress.label')} />
              <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <span>{describeProgress(t, live, job.status)}</span>
                <span aria-hidden="true">{live.progress}%</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={onCancel}
                disabled={isCancelling}
                aria-busy={isCancelling}
              >
                <Ban aria-hidden="true" />
                {isCancelling ? t('job:actions.cancelling') : t('job:actions.cancel')}
              </Button>
            </>
          )}

          {job.status === 'completed' && job.output !== null && (
            <>
              <dl className="flex flex-col gap-1.5 text-xs">
                <Row label={t('job:result.size')}>
                  <MetricBadge>
                    {formatDimensions(job.output.width, job.output.height, locale)}
                  </MetricBadge>
                </Row>
                <Row label={t('job:result.fileSize')}>
                  <MetricBadge>{formatBytes(job.output.sizeBytes, 1, locale)}</MetricBadge>
                </Row>
                {job.processingMs !== null && (
                  <Row label={t('job:result.took')}>
                    <MetricBadge>{formatDuration(job.processingMs, locale)}</MetricBadge>
                  </Row>
                )}
              </dl>
              <Button asChild className="w-full">
                {/* A plain link: the browser saves the file straight from the
                    server, honouring Content-Disposition, without the image
                    passing through JavaScript. */}
                <a href={resultUrl(job.jobId)} download>
                  <Download aria-hidden="true" />
                  {t('job:actions.download')}
                </a>
              </Button>
            </>
          )}

          {job.status === 'failed' && job.error !== null && (
            <ProblemErrorPanel problem={job.error} title={t('job:failedTitle')} />
          )}

          {job.status === 'cancelled' && (
            <p className="text-xs text-muted-foreground">{t('job:cancelledNote')}</p>
          )}

          {isFinished && (
            <Button variant="ghost" size="sm" onClick={onReset}>
              <RotateCcw aria-hidden="true" />
              {t('job:actions.again')}
            </Button>
          )}
        </div>
      )}

      {problem !== undefined && (
        <ProblemErrorPanel
          problem={problem}
          onRetry={onReset}
          retryLabel={t('errors:dismiss')}
        />
      )}
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}
