import { useTranslation } from 'react-i18next'
import { MetricBadge } from '@/components/ui/badge'
import { Field } from '@/components/ui/field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { formatDimensions } from '@/lib/format'
import { QUALITY_RANGE, isLossy, useEnhancementStore } from '@/stores/useEnhancementStore'
import type { ImageMetadata } from '@/types/image'
import type { OutputFormat } from '@/types/job'
import { projectedSize } from './jobPresentation'

/**
 * Output format, quality and metadata handling.
 *
 * Quality is disabled for PNG rather than hidden: a control that vanishes
 * makes the format switch feel unpredictable, and PNG being lossless is worth
 * saying once rather than leaving the user to infer it.
 */

/**
 * The encodings on offer.
 *
 * The labels are format names, identical in every language; only PNG's hint is
 * a word, so only that one is translated.
 */
const FORMAT_OPTIONS = [
  { value: 'png', label: 'PNG', translateHint: true },
  { value: 'jpeg', label: 'JPEG', translateHint: false },
  { value: 'webp', label: 'WEBP', translateHint: false },
] as const

export function OutputControls({ source }: { source: ImageMetadata | undefined }) {
  const { t, i18n } = useTranslation('enhance')
  const format = useEnhancementStore((s) => s.format)
  const setFormat = useEnhancementStore((s) => s.setFormat)
  const quality = useEnhancementStore((s) => s.quality)
  const setQuality = useEnhancementStore((s) => s.setQuality)
  const preserveMetadata = useEnhancementStore((s) => s.preserveMetadata)
  const setPreserveMetadata = useEnhancementStore((s) => s.setPreserveMetadata)
  const scale = useEnhancementStore((s) => s.scale)

  const projected = projectedSize(source, scale)
  const lossy = isLossy(format)

  return (
    <div className="flex flex-col gap-5">
      <Field label={t('format.label')} description={t('format.description')}>
        {({ describedBy }) => (
          <div aria-describedby={describedBy}>
            <SegmentedControl
              name="output-format"
              label={t('format.groupLabel')}
              value={format}
              onChange={(value) => { setFormat(value as OutputFormat) }}
              options={FORMAT_OPTIONS.map((option) => ({
                value: option.value,
                label: option.label,
                ...(option.translateHint ? { hint: t('format.losslessHint') } : {}),
              }))}
            />
          </div>
        )}
      </Field>

      <Field
        label={t('quality.label')}
        description={
          lossy ? t('quality.descriptionLossy') : t('quality.descriptionLossless')
        }
      >
        {({ id, describedBy }) => (
          <div className="flex items-center gap-3">
            <Slider
              id={id}
              aria-describedby={describedBy}
              aria-label={t('quality.sliderLabel')}
              value={[quality]}
              onValueChange={([value]) => { setQuality(value ?? QUALITY_RANGE.max) }}
              min={QUALITY_RANGE.min}
              max={QUALITY_RANGE.max}
              step={1}
              disabled={!lossy}
            />
            <MetricBadge className="w-12 justify-center">{lossy ? quality : '—'}</MetricBadge>
          </div>
        )}
      </Field>

      <Field
        label={t('metadata.label')}
        description={t('metadata.description')}
        orientation="horizontal"
      >
        {({ id, describedBy }) => (
          <Switch
            id={id}
            aria-describedby={describedBy}
            checked={preserveMetadata}
            onCheckedChange={setPreserveMetadata}
          />
        )}
      </Field>

      {projected !== undefined && (
        <dl className="flex items-center justify-between border-t border-border pt-3 text-xs">
          <dt className="text-muted-foreground">{t('projected.label')}</dt>
          <dd>
            <MetricBadge>
              {formatDimensions(projected.width, projected.height, i18n.language)}
            </MetricBadge>
          </dd>
        </dl>
      )}
    </div>
  )
}
