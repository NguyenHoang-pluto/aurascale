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
import { useEnhancementStore } from '@/stores/useEnhancementStore'
import type { ModelInfo } from '@/types/system'
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
      <Field label="Model" description="Which trained weights to run.">
        {({ id, describedBy }) => (
          <SelectRoot
            value={selected?.id ?? ''}
            onValueChange={(value) => {
              const next = models.find((model) => model.id === value)
              if (next !== undefined) setModel(next.id, next.supportedScales)
            }}
          >
            <SelectTrigger id={id} aria-describedby={describedBy}>
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem
                  key={model.id}
                  value={model.id}
                  disabled={!model.downloaded}
                  description={
                    model.downloaded
                      ? model.description
                      : `${model.description} — not downloaded`
                  }
                >
                  {model.name}
                </SelectItem>
              ))}
            </SelectContent>
          </SelectRoot>
        )}
      </Field>

      <Field
        label="Upscale factor"
        description={
          supported.includes(8)
            ? 'Every factor is produced by neural passes. 8x runs two: 4x then 2x.'
            : 'Factors this model cannot produce are unavailable.'
        }
      >
        {({ describedBy }) => (
          <div aria-describedby={describedBy}>
            <SegmentedControl
              name="scale"
              label="Upscale factor"
              value={String(scale)}
              onChange={(value) => { setScale(Number(value)) }}
              options={scaleOptions(supported)}
            />
          </div>
        )}
      </Field>

      <Field
        label="Noise reduction"
        description={
          selected?.supportsDenoise === true
            ? 'Interpolates the model weights. 100% is the model as shipped, which denoises the most; lowering it blends toward weights that keep more grain.'
            : `${selected?.name ?? 'This model'} has no denoise weights to blend.`
        }
      >
        {({ id, describedBy }) => (
          <div className="flex items-center gap-3">
            <Slider
              id={id}
              aria-describedby={describedBy}
              aria-label="Noise reduction"
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
        label="Sharpening"
        description="An unsharp mask applied after upscaling. A post-process, not a model feature."
      >
        {({ id, describedBy }) => (
          <div className="flex items-center gap-3">
            <Slider
              id={id}
              aria-describedby={describedBy}
              aria-label="Sharpening strength"
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
        label="Artifact reduction"
        description="Needs a dedicated JPEG restoration model. A blur filter here would smear detail while claiming to add it."
        orientation="horizontal"
        comingSoon
      >
        {({ id, describedBy }) => (
          <Switch id={id} aria-describedby={describedBy} checked={false} disabled />
        )}
      </Field>

      {selected !== undefined && !selected.downloaded && (
        <p className="flex items-center gap-2 text-xs text-warning">
          <Badge tone="warning">Not downloaded</Badge>
          Run scripts/download_models.py before enhancing with this model.
        </p>
      )}
    </div>
  )
}
