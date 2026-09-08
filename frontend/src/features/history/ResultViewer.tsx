import { Download, X } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { Button } from '@/components/ui/button'
import { MetricBadge } from '@/components/ui/badge'
import { ZoomControls } from '@/features/viewer/ZoomControls'
import { useZoomPan } from '@/features/viewer/useZoomPan'
import { formatBytes, formatDimensions } from '@/lib/format'
import { cn } from '@/lib/cn'
import { previewUrl, resultUrl } from '@/services/jobsApi'
import type { JobRecord } from '@/types/job'

/**
 * A past result, on its own.
 *
 * Not the Phase 9 comparison: that needs the original, which lives only in the
 * browser for the session that submitted it. The server keeps the input file
 * but does not serve it, so a historical job can show what was produced and
 * nothing more. Claiming a before/after here would mean comparing the result
 * against itself.
 *
 * Reuses `useZoomPan` and the capped preview, so a 200 MP result is no more
 * expensive to look at here than in the workspace.
 */
export function ResultViewer({ job, onClose }: { job: JobRecord; onClose: () => void }) {
  const { t, i18n } = useTranslation(['history', 'compare', 'common'])
  const locale = i18n.language
  const output = job.output
  const {
    attachContainer,
    transform,
    isFitted,
    isPanning,
    canPan,
    zoomIn,
    zoomOut,
    setScale,
    fit,
    actualSize,
    handlers,
  } = useZoomPan(output?.width ?? 0, output?.height ?? 0)

  if (output === null) return null

  return (
    <section
      aria-label={t('history:viewer.label')}
      className="flex min-h-0 flex-col overflow-hidden rounded-lg border border-border bg-surface"
    >
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-3 py-2">
        <ZoomControls
          scale={transform.scale}
          isFitted={isFitted}
          onZoomIn={zoomIn}
          onZoomOut={zoomOut}
          onSetScale={setScale}
          onFit={fit}
          onActualSize={actualSize}
        />
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <MetricBadge>{formatDimensions(output.width, output.height, locale)}</MetricBadge>
          <MetricBadge>{formatBytes(output.sizeBytes, 1, locale)}</MetricBadge>
          <Button asChild variant="secondary" size="sm">
            <a href={resultUrl(job.jobId)} download>
              <Download aria-hidden="true" />
              {t('common:actions.download')}
            </a>
          </Button>
          <Button variant="ghost" size="sm" onClick={onClose}>
            <X aria-hidden="true" />
            {t('common:actions.close')}
          </Button>
        </div>
      </div>

      <div
        ref={attachContainer}
        tabIndex={0}
        role="group"
        aria-label={t('history:viewer.surfaceLabel')}
        onPointerDown={handlers.onPointerDown}
        onPointerMove={handlers.onPointerMove}
        onPointerUp={handlers.onPointerUp}
        onPointerCancel={handlers.onPointerUp}
        onKeyDown={handlers.onKeyDown}
        className={cn(
          'relative h-[60vh] min-h-0 overflow-hidden bg-canvas',
          canPan && (isPanning ? 'cursor-grabbing' : 'cursor-grab'),
        )}
      >
        <img
          src={previewUrl(job.jobId)}
          alt={t('compare:layer.enhancedAlt')}
          draggable={false}
          className="absolute top-0 left-0 max-w-none origin-top-left select-none"
          style={{
            width: `${String(output.width)}px`,
            height: `${String(output.height)}px`,
            transform: `translate(${String(transform.tx)}px, ${String(transform.ty)}px) scale(${String(transform.scale)})`,
            imageRendering: transform.scale > 1 ? 'pixelated' : 'auto',
          }}
        />
      </div>
    </section>
  )
}
