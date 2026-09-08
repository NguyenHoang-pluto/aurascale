import { Ban, Download, RotateCcw, Sparkles } from 'lucide-react'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { StatusIndicator, type StatusTone } from '@/components/ui/status'
import { formatBytes, formatDimensions, formatDuration } from '@/lib/format'
import { resultUrl } from '@/services/jobsApi'
import type { ProblemDetail } from '@/types/api'
import type { JobRecord, JobStatus } from '@/types/job'
import { describeProgress, statusLabel } from './jobPresentation'
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
            {isSubmitting ? 'Submitting…' : 'Enhance'}
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
              label={statusLabel(job.status)}
            />
            {job.device !== null && <Badge tone="neutral">{job.device.toUpperCase()}</Badge>}
          </div>

          {isRunning && (
            <>
              <Progress value={live.progress} label="Enhancement progress" />
              <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                <span>{describeProgress(live, job.status)}</span>
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
                {isCancelling ? 'Stopping…' : 'Cancel'}
              </Button>
            </>
          )}

          {job.status === 'completed' && job.output !== null && (
            <>
              <dl className="flex flex-col gap-1.5 text-xs">
                <Row label="Result">
                  <MetricBadge>
                    {formatDimensions(job.output.width, job.output.height)}
                  </MetricBadge>
                </Row>
                <Row label="File size">
                  <MetricBadge>{formatBytes(job.output.sizeBytes)}</MetricBadge>
                </Row>
                {job.processingMs !== null && (
                  <Row label="Took">
                    <MetricBadge>{formatDuration(job.processingMs)}</MetricBadge>
                  </Row>
                )}
              </dl>
              <Button asChild className="w-full">
                {/* A plain link: the browser saves the file straight from the
                    server, honouring Content-Disposition, without the image
                    passing through JavaScript. */}
                <a href={resultUrl(job.jobId)} download>
                  <Download aria-hidden="true" />
                  Download result
                </a>
              </Button>
            </>
          )}

          {job.status === 'failed' && job.error !== null && (
            <ErrorPanel
              title="Enhancement failed"
              detail={job.error.detail}
              code={job.error.code}
              {...(job.error.technical != null ? { technical: job.error.technical } : {})}
            />
          )}

          {job.status === 'cancelled' && (
            <p className="text-xs text-muted-foreground">
              Stopped before it finished, so no result was produced.
            </p>
          )}

          {isFinished && (
            <Button variant="ghost" size="sm" onClick={onReset}>
              <RotateCcw aria-hidden="true" />
              Enhance again
            </Button>
          )}
        </div>
      )}

      {problem !== undefined && (
        <ErrorPanel
          title={problem.title}
          detail={problem.detail}
          code={problem.code}
          {...(problem.technical !== undefined ? { technical: problem.technical } : {})}
          onRetry={onReset}
          retryLabel="Dismiss"
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
