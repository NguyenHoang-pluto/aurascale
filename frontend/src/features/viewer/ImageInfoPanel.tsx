import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { PhaseNotice } from '@/components/feedback/PhaseNotice'
import { formatBytes, formatDimensions, formatMegapixels } from '@/lib/format'
import type { ImageMetadata } from '@/types/image'

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-mono text-xs tabular-nums text-foreground">{value}</dd>
    </div>
  )
}

/**
 * Image information (§ 8).
 *
 * Only the "Original" side can be populated today. Rather than showing an
 * "Enhanced" column full of dashes — which reads as broken — the result side
 * states which phase produces it.
 */
export function ImageInfoPanel({ metadata }: { metadata: ImageMetadata }) {
  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>Image information</PanelTitle>
      </PanelHeader>
      <PanelContent>
        <p className="mb-1 text-xs font-medium text-foreground">Original</p>
        <dl>
          <InfoRow
            label="Dimensions"
            value={formatDimensions(metadata.width, metadata.height)}
          />
          <InfoRow
            label="Resolution"
            value={formatMegapixels(metadata.width, metadata.height)}
          />
          <InfoRow label="File size" value={formatBytes(metadata.sizeBytes)} />
          <InfoRow label="Format" value={metadata.format} />
        </dl>

        <PhaseNotice
          title="Enhanced"
          description="Output dimensions, file size, format, upscale factor, processing time and model appear here once a job has run."
          phase="Phase 7"
          className="mt-3 px-3 py-4"
        />
      </PanelContent>
    </Panel>
  )
}
