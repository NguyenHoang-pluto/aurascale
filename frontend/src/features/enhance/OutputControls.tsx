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

const FORMAT_OPTIONS = [
  { value: 'png', label: 'PNG', hint: 'lossless' },
  { value: 'jpeg', label: 'JPEG' },
  { value: 'webp', label: 'WEBP' },
] as const

export function OutputControls({ source }: { source: ImageMetadata | undefined }) {
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
      <Field label="Format" description="How the result is encoded.">
        {({ describedBy }) => (
          <div aria-describedby={describedBy}>
            <SegmentedControl
              name="output-format"
              label="Output format"
              value={format}
              onChange={(value) => { setFormat(value as OutputFormat) }}
              options={FORMAT_OPTIONS.map((option) => ({ ...option }))}
            />
          </div>
        )}
      </Field>

      <Field
        label="Quality"
        description={
          lossy
            ? 'Higher keeps more detail and makes a larger file.'
            : 'PNG is lossless, so there is no quality to trade.'
        }
      >
        {({ id, describedBy }) => (
          <div className="flex items-center gap-3">
            <Slider
              id={id}
              aria-describedby={describedBy}
              aria-label="Output quality"
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
        label="Preserve metadata"
        description="Keep EXIF and the ICC colour profile. Turning this off drops location and camera tags."
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
          <dt className="text-muted-foreground">Result size</dt>
          <dd>
            <MetricBadge>{formatDimensions(projected.width, projected.height)}</MetricBadge>
          </dd>
        </dl>
      )}
    </div>
  )
}
