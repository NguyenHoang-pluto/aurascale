import { useTranslation } from 'react-i18next'
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
  const { t, i18n } = useTranslation('viewer')
  const locale = i18n.language

  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>{t('info.title')}</PanelTitle>
      </PanelHeader>
      <PanelContent>
        <p className="mb-1 text-xs font-medium text-foreground">{t('info.original')}</p>
        <dl>
          <InfoRow
            label={t('info.dimensions')}
            value={formatDimensions(metadata.width, metadata.height, locale)}
          />
          <InfoRow
            label={t('info.resolution')}
            value={formatMegapixels(metadata.width, metadata.height, locale)}
          />
          <InfoRow label={t('info.fileSize')} value={formatBytes(metadata.sizeBytes, 1, locale)} />
          {/* A format name is an identifier: PNG is PNG in both languages. */}
          <InfoRow label={t('info.format')} value={metadata.format} />
        </dl>

        <PhaseNotice
          title={t('info.enhancedTitle')}
          description={t('info.enhancedDescription')}
          phase="Phase 7"
          className="mt-3 px-3 py-4"
        />
      </PanelContent>
    </Panel>
  )
}
