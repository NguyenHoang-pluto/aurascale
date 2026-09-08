import { useTranslation } from 'react-i18next'
import { SegmentedControl } from '@/components/ui/segmented-control'
import type { CompareMode } from './comparisonMath'

/**
 * The three ways of looking at a before and an after.
 *
 * The value is the mode the viewer works in and never changes with language;
 * the key is what the user reads.
 */
const MODES = [
  { value: 'slider', key: 'mode.slider', hintKey: 'mode.sliderHint' },
  { value: 'side-by-side', key: 'mode.sideBySide', hintKey: null },
  { value: 'split', key: 'mode.split', hintKey: null },
] as const

export function CompareModeControl({
  mode,
  onChange,
}: {
  mode: CompareMode
  onChange: (mode: CompareMode) => void
}) {
  const { t } = useTranslation('compare')

  return (
    <SegmentedControl
      name="compare-mode"
      label={t('mode.label')}
      value={mode}
      onChange={(value) => { onChange(value as CompareMode) }}
      options={MODES.map((option) => ({
        value: option.value,
        label: t(option.key),
        ...(option.hintKey === null ? {} : { hint: t(option.hintKey) }),
      }))}
      size="sm"
      className="w-auto"
    />
  )
}
