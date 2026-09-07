import { Maximize2, Minus, Plus, Scan } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Tooltip } from '@/components/ui/tooltip'
import { MAX_SCALE, MIN_SCALE, ZOOM_PRESETS } from './zoomMath'

export function ZoomControls({
  scale,
  isFitted,
  onZoomIn,
  onZoomOut,
  onSetScale,
  onFit,
  onActualSize,
}: {
  scale: number
  isFitted: boolean
  onZoomIn: () => void
  onZoomOut: () => void
  onSetScale: (scale: number) => void
  onFit: () => void
  onActualSize: () => void
}) {
  const percent = Math.round(scale * 100)

  return (
    <div className="flex flex-wrap items-center gap-1">
      <Tooltip content="Zoom out (-)">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Zoom out"
          disabled={scale <= MIN_SCALE + 1e-6}
          onClick={onZoomOut}
        >
          <Minus aria-hidden="true" />
        </Button>
      </Tooltip>

      {/*
        A native select rather than a custom menu: it is one tab stop, works
        with a keyboard and on touch, and needs no popover management. The
        current zoom may not be a preset (wheel zoom is continuous), so the
        live value is offered as an extra option rather than being snapped.
      */}
      <label className="sr-only" htmlFor="zoom-level">
        Zoom level
      </label>
      <select
        id="zoom-level"
        value={String(percent)}
        onChange={(event) => { onSetScale(Number(event.target.value) / 100) }}
        className={
          'h-8 rounded-md border border-border bg-surface-raised px-2 ' +
          'font-mono text-xs tabular-nums text-foreground'
        }
      >
        {!ZOOM_PRESETS.some((preset) => Math.round(preset * 100) === percent) && (
          <option value={String(percent)}>{percent}%</option>
        )}
        {ZOOM_PRESETS.map((preset) => (
          <option key={preset} value={String(Math.round(preset * 100))}>
            {Math.round(preset * 100)}%
          </option>
        ))}
      </select>

      <Tooltip content="Zoom in (+)">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Zoom in"
          disabled={scale >= MAX_SCALE - 1e-6}
          onClick={onZoomIn}
        >
          <Plus aria-hidden="true" />
        </Button>
      </Tooltip>

      <Separator orientation="vertical" className="mx-1 h-5" />

      <Tooltip content="Fit to screen (0)">
        <Button
          variant={isFitted ? 'secondary' : 'ghost'}
          size="icon-sm"
          aria-label="Fit to screen"
          aria-pressed={isFitted}
          onClick={onFit}
        >
          <Scan aria-hidden="true" />
        </Button>
      </Tooltip>

      <Tooltip content="Actual size, 100% (1)">
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Actual size"
          onClick={onActualSize}
        >
          <Maximize2 aria-hidden="true" />
        </Button>
      </Tooltip>
    </div>
  )
}
