import { Download } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { Button } from '@/components/ui/button'
import { resultUrl } from '@/services/jobsApi'
import { ComparisonViewer } from './ComparisonViewer'
import type { CompletedResult } from './useCompletedResult'

/**
 * The comparison surface for a finished job.
 *
 * Rendered by the page only when `useCompletedResult` says there is something
 * to compare, so this component never has to reason about queued, running,
 * failed or cancelled jobs.
 */
export function ResultCanvas({
  result,
  toolbarSlot,
}: {
  result: CompletedResult
  toolbarSlot?: React.ReactNode
}) {
  const { t } = useTranslation('compare')
  const [previewFailed, setPreviewFailed] = useState(false)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ComparisonViewer
        jobId={result.jobId}
        before={result.before}
        output={result.output}
        previewFailed={previewFailed}
        onPreviewError={() => { setPreviewFailed(true) }}
        toolbarSlot={toolbarSlot}
      />

      {previewFailed && (
        <div className="flex shrink-0 flex-col gap-3 border-t border-border p-4">
          {/* No `code`. This panel is driven by an <img> onError, which tells
              us the browser could not load the preview and nothing else - not
              the status, not a problem document. It previously claimed
              "job_not_found", which sent a real investigation looking for a
              deleted job that was never deleted. Saying nothing is better than
              naming a cause we cannot know. */}
          <ErrorPanel
            title={t('previewFailed.title')}
            detail={t('previewFailed.detail')}
            onRetry={() => { setPreviewFailed(false) }}
          />
          {/* Download stays reachable: a failed preview says nothing about the
              result file, which is served by a different endpoint. */}
          <Button asChild variant="secondary" size="sm" className="self-start">
            <a href={resultUrl(result.jobId)} download>
              <Download aria-hidden="true" />
              {t('previewFailed.download')}
            </a>
          </Button>
        </div>
      )}
    </div>
  )
}
