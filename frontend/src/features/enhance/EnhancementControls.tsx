import { useTranslation } from 'react-i18next'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Field } from '@/components/ui/field'
import {
  SelectContent,
  SelectItem,
  SelectRoot,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Skeleton } from '@/components/ui/skeleton'
import { Slider } from '@/components/ui/slider'
import { Switch } from '@/components/ui/switch'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { useModelDescription } from '@/i18n/modelMessages'
import { useEnhancementStore } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import type { ModelInfo } from '@/types/system'
import { ModeControls } from './ModeControls'
import { scaleOptions } from './jobPresentation'

/**
 * Model, scale and the two strength controls.
 *
 * Which factors a model can produce is not decided here: `/api/models`
 * publishes `supportedScales`, derived on the server from the same planner
 * that validates a job. Anything absent from that list is rendered disabled,
 * so an impossible combination is visible but unselectable rather than
 * submitted and rejected.
 *
 * A factor the loaded image is simply too large for - 16x of a 12 MP photo is
 * 3 gigapixels - is disabled the same way, judged against the same output
 * limit the backend enforces. Both remain re-checked on the server.
 *
 * The labelling follows docs/image-processing.md § 7: model and scale are AI,
 * noise reduction is AI (weight interpolation), sharpening is a post-process
 * and says so, and artifact reduction is visibly unbuilt rather than faked.
 */

export function EnhancementControls({
  models,
  isLoading,
  problem,
  selected,
}: {
  models: readonly ModelInfo[]
  isLoading: boolean
  problem?: { title: string; detail: string; code?: string; technical?: string }
  /** The currently selected model, if the registry has loaded. */
  selected: ModelInfo | undefined
}) {
  const { t } = useTranslation(['enhance', 'common'])
  const describeModel = useModelDescription()
  const source = useWorkspaceStore((s) => s.source)
  const sizing = useEnhancementStore((s) => s.sizing)
  const scale = useEnhancementStore((s) => s.scale)
  const setScale = useEnhancementStore((s) => s.setScale)
  const setModel = useEnhancementStore((s) => s.setModel)
  const denoiseStrength = useEnhancementStore((s) => s.denoiseStrength)
  const setDenoiseStrength = useEnhancementStore((s) => s.setDenoiseStrength)
  const sharpenStrength = useEnhancementStore((s) => s.sharpenStrength)
  const setSharpenStrength = useEnhancementStore((s) => s.setSharpenStrength)

  if (problem !== undefined) {
    return (
      <ErrorPanel
        title={problem.title}
        detail={problem.detail}
        {...(problem.code !== undefined ? { code: problem.code } : {})}
        {...(problem.technical !== undefined ? { technical: problem.technical } : {})}
        className="border-0 bg-transparent p-0"
      />
    )
  }

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4" aria-busy="true">
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
        <Skeleton className="h-9 w-full" />
      </div>
    )
  }

  const supported = selected?.supportedScales ?? []

  return (
    <div className="flex flex-col gap-5">
      <ModeControls supportedScales={supported} />

      <Field label={t('enhance:model.label')} description={t('enhance:model.description')}>
        {({ id, describedBy }) => (
          <SelectRoot
            value={selected?.id ?? ''}
            onValueChange={(value) => {
              const next = models.find((model) => model.id === value)
              if (next !== undefined) setModel(next.id, next.supportedScales)
            }}
          >
            <SelectTrigger id={id} aria-describedby={describedBy}>
              <SelectValue placeholder={t('enhance:model.placeholder')} />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem
                  key={model.id}
                  value={model.id}
                  disabled={!model.downloaded}
                  description={
                    model.downloaded
                      ? describeModel(model)
                      : t('enhance:model.notDownloadedSuffix', {
                          description: describeModel(model),
                        })
                  }
                >
                  {/* The name is an identifier — it matches the manifest and
                      the weights file, so it reads the same in every language. */}
                  {model.name}
                </SelectItem>
              ))}
            </SelectContent>
          </SelectRoot>
        )}
      </Field>

      <Field
        label={t('enhance:scale.label')}
        description={
          supported.includes(8)
            ? t('enhance:scale.descriptionTwoPass')
            : t('enhance:scale.descriptionLimited')
        }
      >
        {({ describedBy }) => (
          <div aria-describedby={describedBy}>
            <SegmentedControl
              name="scale"
              label={t('enhance:scale.label')}
              value={String(scale)}
              onChange={(value) => { setScale(Number(value)) }}
              // A target already decides the size, so every factor is
              // disabled while one is chosen. The control stays visible rather
              // than vanishing, so the two read as alternatives rather than
              // one silently replacing the other.
              options={scaleOptions(t, supported, source?.metadata).map((option) => ({
                ...option,
                disabled: option.disabled === true || sizing.kind === 'target',
              }))}
            />
          </div>
        )}
      </Field>

      <Field
        label={t('enhance:denoise.label')}
        description={
          selected?.supportsDenoise === true
            ? t('enhance:denoise.description')
            : selected === undefined
              ? t('enhance:denoise.unavailableGeneric')
              : t('enhance:denoise.unavailable', { model: selected.name })
        }
      >
        {({ id, describedBy }) => (
          <div className="flex items-center gap-3">
            <Slider
              id={id}
              aria-describedby={describedBy}
              aria-label={t('enhance:denoise.label')}
              value={[Math.round(denoiseStrength * 100)]}
              onValueChange={([value]) => { setDenoiseStrength((value ?? 0) / 100) }}
              max={100}
              step={5}
              disabled={selected?.supportsDenoise !== true}
            />
            <MetricBadge className="w-12 justify-center">
              {selected?.supportsDenoise === true
                ? `${String(Math.round(denoiseStrength * 100))}%`
                : '—'}
            </MetricBadge>
          </div>
        )}
      </Field>

      <Field
        label={t('enhance:sharpen.label')}
        description={t('enhance:sharpen.description')}
      >
        {({ id, describedBy }) => (
          <div className="flex items-center gap-3">
            <Slider
              id={id}
              aria-describedby={describedBy}
              aria-label={t('enhance:sharpen.sliderLabel')}
              value={[Math.round(sharpenStrength * 100)]}
              onValueChange={([value]) => { setSharpenStrength((value ?? 0) / 100) }}
              max={100}
              step={5}
            />
            <MetricBadge className="w-12 justify-center">
              {Math.round(sharpenStrength * 100)}%
            </MetricBadge>
          </div>
        )}
      </Field>

      <Field
        label={t('enhance:artifacts.label')}
        description={t('enhance:artifacts.description')}
        orientation="horizontal"
        comingSoon
      >
        {({ id, describedBy }) => (
          <Switch id={id} aria-describedby={describedBy} checked={false} disabled />
        )}
      </Field>

      {selected !== undefined && !selected.downloaded && (
        <p className="flex items-center gap-2 text-xs text-warning">
          <Badge tone="warning">{t('enhance:model.notDownloadedBadge')}</Badge>
          {t('enhance:model.notDownloadedHint')}
        </p>
      )}
    </div>
  )
}
