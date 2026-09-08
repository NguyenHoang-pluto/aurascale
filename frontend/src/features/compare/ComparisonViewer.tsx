import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import { cn } from '@/lib/cn'
import { ZoomControls } from '@/features/viewer/ZoomControls'
import { useZoomPan } from '@/features/viewer/useZoomPan'
import type { Size } from '@/features/viewer/zoomMath'
import { previewUrl } from '@/services/jobsApi'
import { useComparisonStore } from '@/stores/useComparisonStore'
import type { BeforeSnapshot } from '@/stores/useComparisonStore'
import { CompareModeControl } from './CompareModeControl'
import { SPLIT_POSITION, dividerForKey, dividerFromPointer } from './comparisonMath'
import { useCropLayer } from './useCropLayer'

/**
 * Before and after, in one frame or two.
 *
 * The comparison works in **output pixel space**. The enhanced result defines
 * the coordinate system and the original is stretched into the same frame, so
 * at any divider position both sides show the same region of the same subject.
 * Comparing them at their own sizes would put different parts of the picture
 * on either side of the divider, which is not a comparison.
 *
 * One `useZoomPan` instance drives every layer and both panes. Two instances
 * would drift apart the moment either was panned, and a comparison that is not
 * aligned is worse than no comparison.
 *
 * The after layer is the server's capped preview, never the full result. Above
 * 100 % a crop of the visible region is fetched at full resolution and drawn
 * over it (see `useCropLayer`).
 */

export interface ComparisonViewerProps {
  jobId: string
  /** The original, snapshotted at submission time. */
  before: BeforeSnapshot
  /** The result's dimensions, which define the shared coordinate space. */
  output: { width: number; height: number }
  className?: string
  toolbarSlot?: React.ReactNode
  /** Shown instead of the after layer when the preview cannot be loaded. */
  onPreviewError?: () => void
  previewFailed?: boolean
}

export function ComparisonViewer({
  jobId,
  before,
  output,
  className,
  toolbarSlot,
  onPreviewError,
  previewFailed = false,
}: ComparisonViewerProps) {
  const mode = useComparisonStore((s) => s.mode)
  const setMode = useComparisonStore((s) => s.setMode)
  const divider = useComparisonStore((s) => s.divider)
  const setDivider = useComparisonStore((s) => s.setDivider)

  const [isPreviewLoading, setIsPreviewLoading] = useState(true)

  const {
    attachContainer,
    transform,
    isFitted,
    isPanning,
    canPan,
    zoomIn,
    zoomOut,
    setScale,
    fit,
    actualSize,
    handlers,
  } = useZoomPan(output.width, output.height)

  const surfaceRef = useRef<HTMLDivElement | null>(null)
  const containerSize = useContainerSize(surfaceRef)

  const crop = useCropLayer(jobId, transform, containerSize, output, {
    enabled: !previewFailed,
  })

  const attach = useCallback(
    (node: HTMLDivElement | null) => {
      surfaceRef.current = node
      attachContainer(node)
    },
    [attachContainer],
  )

  const position = mode === 'split' ? SPLIT_POSITION : divider

  return (
    <div className={cn('flex min-h-0 flex-1 flex-col', className)}>
      <div className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-border px-3 py-2">
        <ZoomControls
          scale={transform.scale}
          isFitted={isFitted}
          onZoomIn={zoomIn}
          onZoomOut={zoomOut}
          onSetScale={setScale}
          onFit={fit}
          onActualSize={actualSize}
        />
        <CompareModeControl mode={mode} onChange={setMode} />
        {toolbarSlot}
      </div>

      <div
        ref={attach}
        tabIndex={0}
        role="group"
        aria-label="Comparison viewer. Plus and minus zoom, 0 fits to screen, 1 shows actual size, arrow keys pan."
        onPointerDown={handlers.onPointerDown}
        onPointerMove={handlers.onPointerMove}
        onPointerUp={handlers.onPointerUp}
        onPointerCancel={handlers.onPointerUp}
        onKeyDown={handlers.onKeyDown}
        className={cn(
          'relative min-h-0 flex-1 overflow-hidden bg-canvas',
          canPan && (isPanning ? 'cursor-grabbing' : 'cursor-grab'),
        )}
      >
        {mode === 'side-by-side' ? (
          <SideBySide
            before={before}
            jobId={jobId}
            output={output}
            transform={transform}
            crop={crop}
            previewFailed={previewFailed}
            onPreviewLoad={() => { setIsPreviewLoading(false) }}
            onPreviewError={onPreviewError}
          />
        ) : (
          <Wipe
            before={before}
            jobId={jobId}
            output={output}
            transform={transform}
            position={position}
            crop={crop}
            previewFailed={previewFailed}
            onPreviewLoad={() => { setIsPreviewLoading(false) }}
            onPreviewError={onPreviewError}
          />
        )}

        {mode !== 'side-by-side' && (
          <Divider
            position={position}
            draggable={mode === 'slider'}
            onMove={setDivider}
            surfaceRef={surfaceRef}
          />
        )}

        {(isPreviewLoading || crop.isLoading) && !previewFailed && (
          <p
            role="status"
            aria-live="polite"
            className="absolute top-2 right-2 rounded-md border border-border bg-surface-raised/90 px-2 py-1 text-xs text-muted-foreground"
          >
            {isPreviewLoading ? 'Loading result…' : 'Loading detail…'}
          </p>
        )}
      </div>
    </div>
  )
}

/** A layer drawn in output-pixel space, positioned by the shared transform. */
function Layer({
  src,
  alt,
  output,
  transform,
  onLoad,
  onError,
  className,
  style,
}: {
  src: string
  alt: string
  output: { width: number; height: number }
  transform: { scale: number; tx: number; ty: number }
  onLoad?: () => void
  onError?: () => void
  className?: string
  style?: React.CSSProperties
}) {
  return (
    <img
      src={src}
      alt={alt}
      // Laid out at the output's size whatever the file's own resolution is:
      // the preview is smaller and the original is a different size, and both
      // have to occupy the same frame for the comparison to mean anything.
      width={output.width}
      height={output.height}
      draggable={false}
      onLoad={onLoad}
      onError={onError}
      className={cn(
        'absolute top-0 left-0 max-w-none origin-top-left select-none',
        className,
      )}
      style={{
        width: `${String(output.width)}px`,
        height: `${String(output.height)}px`,
        transform: `translate(${String(transform.tx)}px, ${String(transform.ty)}px) scale(${String(transform.scale)})`,
        imageRendering: transform.scale > 1 ? 'pixelated' : 'auto',
        ...style,
      }}
    />
  )
}

/** The full-resolution crop, drawn over the preview where it belongs. */
function CropOverlay({
  url,
  region,
  transform,
}: {
  url: string
  region: { x: number; y: number; w: number; h: number }
  transform: { scale: number; tx: number; ty: number }
}) {
  return (
    <img
      src={url}
      alt=""
      aria-hidden="true"
      draggable={false}
      className="absolute top-0 left-0 max-w-none origin-top-left select-none"
      style={{
        width: `${String(region.w)}px`,
        height: `${String(region.h)}px`,
        transform:
          `translate(${String(transform.tx + region.x * transform.scale)}px, ` +
          `${String(transform.ty + region.y * transform.scale)}px) scale(${String(transform.scale)})`,
        imageRendering: transform.scale > 1 ? 'pixelated' : 'auto',
      }}
    />
  )
}

interface PaneProps {
  before: BeforeSnapshot
  jobId: string
  output: { width: number; height: number }
  transform: { scale: number; tx: number; ty: number }
  crop: { url: string | null; region: { x: number; y: number; w: number; h: number } | null }
  previewFailed: boolean
  onPreviewLoad: () => void
  onPreviewError?: (() => void) | undefined
}

/** Slider and split: one frame, the after clipped at the divider. */
function Wipe({ position, ...props }: PaneProps & { position: number }) {
  const { before, jobId, output, transform, crop, previewFailed, onPreviewLoad, onPreviewError } =
    props

  return (
    <>
      <Layer
        src={before.objectUrl}
        alt={`Original: ${before.name}`}
        output={output}
        transform={transform}
      />

      {!previewFailed && (
        <div
          className="absolute inset-0 overflow-hidden"
          // The clip is in screen space, so the divider stays where the user
          // put it while zooming and panning underneath it.
          style={{ clipPath: `inset(0 ${String(100 - position)}% 0 0)` }}
        >
          <Layer
            src={previewUrl(jobId)}
            alt="Enhanced result"
            output={output}
            transform={transform}
            onLoad={onPreviewLoad}
            {...(onPreviewError !== undefined ? { onError: onPreviewError } : {})}
          />
          {crop.url !== null && crop.region !== null && (
            <CropOverlay url={crop.url} region={crop.region} transform={transform} />
          )}
        </div>
      )}
    </>
  )
}

/** Two panes, one transform. Panning either pans both, because there is one. */
function SideBySide(props: PaneProps) {
  const { before, jobId, output, transform, crop, previewFailed, onPreviewLoad, onPreviewError } =
    props

  return (
    <div className="absolute inset-0 flex">
      <section aria-label="Original" className="relative min-w-0 flex-1 overflow-hidden">
        <Layer
          src={before.objectUrl}
          alt={`Original: ${before.name}`}
          output={output}
          transform={transform}
        />
      </section>

      <div aria-hidden="true" className="w-px shrink-0 bg-border" />

      <section aria-label="Enhanced" className="relative min-w-0 flex-1 overflow-hidden">
        {!previewFailed && (
          <>
            <Layer
              src={previewUrl(jobId)}
              alt="Enhanced result"
              output={output}
              transform={transform}
              onLoad={onPreviewLoad}
              {...(onPreviewError !== undefined ? { onError: onPreviewError } : {})}
            />
            {crop.url !== null && crop.region !== null && (
              <CropOverlay url={crop.url} region={crop.region} transform={transform} />
            )}
          </>
        )}
      </section>
    </div>
  )
}

/**
 * The divider.
 *
 * A real slider by role, so a keyboard user can move it and a screen reader
 * announces where it is. In split mode it is a static marker: the mode exists
 * to hold the comparison still at the halfway point.
 */
function Divider({
  position,
  draggable,
  onMove,
  surfaceRef,
}: {
  position: number
  draggable: boolean
  onMove: (position: number) => void
  surfaceRef: React.RefObject<HTMLDivElement | null>
}) {
  const isDragging = useRef(false)

  const moveTo = useCallback(
    (clientX: number) => {
      const bounds = surfaceRef.current?.getBoundingClientRect()
      if (bounds === undefined) return
      onMove(dividerFromPointer(clientX, bounds))
    },
    [onMove, surfaceRef],
  )

  const onKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLDivElement>) => {
      const next = dividerForKey(event.key, position, { coarse: event.shiftKey })
      if (next === null) return
      // Stops the surface's own arrow-key panning from firing as well: while
      // the divider has focus, the arrows belong to it.
      event.preventDefault()
      event.stopPropagation()
      onMove(next)
    },
    [onMove, position],
  )

  return (
    <div
      className="pointer-events-none absolute inset-y-0"
      style={{ left: `${String(position)}%` }}
    >
      <div className="absolute inset-y-0 -left-px w-0.5 bg-accent/80" />

      {draggable && (
        <div
          role="slider"
          aria-label="Comparison divider"
          aria-valuenow={Math.round(position)}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuetext={`${String(Math.round(position))}% enhanced`}
          tabIndex={0}
          onKeyDown={onKeyDown}
          onPointerDown={(event) => {
            event.stopPropagation()
            isDragging.current = true
            if (typeof event.currentTarget.setPointerCapture === 'function') {
              event.currentTarget.setPointerCapture(event.pointerId)
            }
          }}
          onPointerMove={(event) => {
            if (!isDragging.current) return
            event.stopPropagation()
            moveTo(event.clientX)
          }}
          onPointerUp={(event) => {
            isDragging.current = false
            if (
              typeof event.currentTarget.hasPointerCapture === 'function' &&
              event.currentTarget.hasPointerCapture(event.pointerId)
            ) {
              event.currentTarget.releasePointerCapture(event.pointerId)
            }
          }}
          className={cn(
            'pointer-events-auto absolute top-1/2 -left-4 size-8 -translate-y-1/2',
            'grid cursor-ew-resize place-items-center rounded-full',
            'border border-border bg-surface-raised shadow-sm',
            'focus-visible:ring-2 focus-visible:ring-accent focus-visible:outline-none',
          )}
        >
          <span aria-hidden="true" className="text-xs text-muted-foreground">
            ⇹
          </span>
        </div>
      )}
    </div>
  )
}

/**
 * The surface's size, so the crop layer knows what is actually on screen.
 *
 * A second observer on the node `useZoomPan` also watches, rather than
 * changing that hook's public shape: it is Phase 4 code with its own tests,
 * and an extra ResizeObserver on one element is cheap.
 */
function useContainerSize(ref: React.RefObject<HTMLDivElement | null>): Size {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })

  useLayoutEffect(() => {
    const node = ref.current
    if (node === null) return

    const measure = (width: number, height: number) => {
      setSize((previous) =>
        previous.width === width && previous.height === height
          ? previous
          : { width, height },
      )
    }

    const rect = node.getBoundingClientRect()
    measure(rect.width, rect.height)

    if (typeof ResizeObserver === 'undefined') return

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry === undefined) return
      measure(entry.contentRect.width, entry.contentRect.height)
    })
    observer.observe(node)

    return () => { observer.disconnect() }
  }, [ref])

  return size
}
