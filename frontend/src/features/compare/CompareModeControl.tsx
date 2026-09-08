import { SegmentedControl } from '@/components/ui/segmented-control'
import type { CompareMode } from './comparisonMath'

/** The three ways of looking at a before and an after. */
const MODES = [
  { value: 'slider', label: 'Slider', hint: 'drag' },
  { value: 'side-by-side', label: 'Side by side' },
  { value: 'split', label: 'Split' },
] as const

export function CompareModeControl({
  mode,
  onChange,
}: {
  mode: CompareMode
  onChange: (mode: CompareMode) => void
}) {
  return (
    <SegmentedControl
      name="compare-mode"
      label="Comparison mode"
      value={mode}
      onChange={(value) => { onChange(value as CompareMode) }}
      options={MODES.map((option) => ({ ...option }))}
      size="sm"
      className="w-auto"
    />
  )
}
