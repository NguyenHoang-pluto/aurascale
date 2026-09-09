import { useTranslation } from 'react-i18next'
import { Field } from '@/components/ui/field'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { useEnhancementStore } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import type { EnhancementMode, TargetResolution } from '@/types/job'
import { TARGET_LONG_EDGE } from '@/types/job'
import { ALL_TARGETS, planTargets } from './targetPlanning'
import type { TargetAvailability } from './targetPlanning'

/**
 * Mode, and how the output size is expressed.
 *
 * Both are deliberately intent-shaped rather than implementation-shaped. Mode
 * says what to prioritise and the backend decides which network that means; no
 * model id appears here, because "realesr-general-x4v3" is not a thing to ask
 * a person to choose between. The model dropdown below still exists for anyone
 * who wants it, and an explicit choice there overrides the mode.
 *
 * Size is a factor *or* a destination, never both. They answer the same
 * question two different ways and the backend refuses the pair, so the control
 * is one selector rather than two that can contradict each other.
 *
 * A preset the loaded image cannot reach is disabled with the reason on hover,
 * rather than offered and then refused after the upload. The backend re-checks
 * regardless; this only saves the user the round trip.
 */

const MODES: readonly { value: EnhancementMode; key: string; hint: string }[] = [
  { value: 'standard', key: 'mode.standard', hint: 'mode.standardHint' },
  { value: 'creative', key: 'mode.creative', hint: 'mode.creativeHint' },
]

/** Everything selectable, for when no image is loaded to judge against. */
const UNJUDGED: TargetAvailability[] = ALL_TARGETS.map((target) => ({
  target,
  available: true,
}))

export function ModeControls({
  supportedScales = [],
}: {
  /** What the selected model offers. A preset needing more is unreachable. */
  supportedScales?: readonly number[]
}) {
  const { t } = useTranslation('enhance')
  const source = useWorkspaceStore((s) => s.source)
  const mode = useEnhancementStore((s) => s.mode)
  const setMode = useEnhancementStore((s) => s.setMode)
  const sizing = useEnhancementStore((s) => s.sizing)
  const setSizing = useEnhancementStore((s) => s.setSizing)

  const selected = MODES.find((option) => option.value === mode) ?? MODES[0]!
  const sizingValue = sizing.kind === 'scale' ? 'scale' : sizing.target

  // Availability depends on the loaded image, so with nothing loaded every
  // preset stays selectable and the backend answers if one turns out not to be.
  const targets =
    source === null
      ? UNJUDGED
      : planTargets(source.metadata.width, source.metadata.height, supportedScales)

  return (
    <>
      <Field label={t('mode.label')} description={t(selected.hint)}>
        {({ describedBy }) => (
          <div aria-describedby={describedBy}>
            <SegmentedControl
              name="enhancement-mode"
              label={t('mode.label')}
              value={mode}
              onChange={(value) => { setMode(value as EnhancementMode) }}
              options={MODES.map((option) => ({
                value: option.value,
                label: t(option.key),
              }))}
            />
          </div>
        )}
      </Field>

      <Field
        label={t('sizing.label')}
        description={
          sizing.kind === 'scale'
            ? t('sizing.descriptionScale')
            : t('sizing.descriptionTarget')
        }
      >
        {({ id, describedBy }) => (
          <select
            id={id}
            aria-describedby={describedBy}
            value={sizingValue}
            onChange={(event) => {
              const next = event.target.value
              setSizing(
                next === 'scale'
                  ? { kind: 'scale' }
                  : { kind: 'target', target: next as TargetResolution },
              )
            }}
            className={
              'h-9 w-full rounded-md border border-border bg-surface-raised px-3 ' +
              'text-sm text-foreground transition-colors hover:bg-muted'
            }
          >
            <option value="scale">{t('sizing.byScale')}</option>
            {targets.map((entry) => (
              // The pixel count is shown because "2K" means 2048 to a cinema
              // and 2560 to a gamer, and neither reading is wrong.
              <option
                key={entry.target}
                value={entry.target}
                disabled={!entry.available}
                {...(entry.reason !== undefined
                  ? { title: t(`sizing.unavailable.${entry.reason}`) }
                  : {})}
              >
                {t('sizing.targetEdge', {
                  label: entry.target.toUpperCase(),
                  pixels: TARGET_LONG_EDGE[entry.target],
                })}
                {entry.available ? '' : ` — ${t('sizing.unavailableSuffix')}`}
              </option>
            ))}
          </select>
        )}
      </Field>
    </>
  )
}
