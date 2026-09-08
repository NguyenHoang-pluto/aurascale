import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/cn'
import type { LoadedImage } from '@/types/image'
import { useZoomPan } from './useZoomPan'
import { ZoomControls } from './ZoomControls'

/**
 * Single-image viewer with zoom and pan.
 *
 * The image is positioned with a CSS transform rather than by changing its
 * width/height: transforms are composited on the GPU, so panning a 16 MP image
 * stays smooth, and the browser never re-rasterises the bitmap mid-drag.
 *
 * The before/after comparison replaces this surface in Phase 9; it reuses the
 * same zoom/pan model so the two views stay in sync.
 */
export function ImageViewer({
  image,
  className,
  toolbarSlot,
}: {
  image: LoadedImage
  className?: string
  /** Rendered beside the zoom controls, e.g. the file name. */
  toolbarSlot?: React.ReactNode
}) {
  const { t } = useTranslation('viewer')
  const { metadata } = image
  // Destructured at the call site: the React Compiler lint treats repeated
  // member access on a hook result that carries a callback ref as accessing a
  // ref during render.
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
  } = useZoomPan(metadata.width, metadata.height)

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
        {toolbarSlot}
      </div>

      <div
        ref={attachContainer}
        // Focusable so the keyboard shortcuts work without a pointer. The
        // shortcuts are named in the label because they are not discoverable
        // any other way.
        tabIndex={0}
        role="group"
        aria-label={t('surface.label')}
        onPointerDown={handlers.onPointerDown}
        onPointerMove={handlers.onPointerMove}
        onPointerUp={handlers.onPointerUp}
        onPointerCancel={handlers.onPointerUp}
        onKeyDown={handlers.onKeyDown}
        className={cn(
          'relative min-h-0 flex-1 overflow-hidden bg-canvas',
          // The checkerboard shows through transparent regions, so a PNG with
          // alpha is not mistaken for one with a white background.
          '[background-image:linear-gradient(45deg,var(--color-muted)_25%,transparent_25%),linear-gradient(-45deg,var(--color-muted)_25%,transparent_25%),linear-gradient(45deg,transparent_75%,var(--color-muted)_75%),linear-gradient(-45deg,transparent_75%,var(--color-muted)_75%)]',
          '[background-size:16px_16px]',
          '[background-position:0_0,0_8px,8px_-8px,-8px_0]',
          canPan && (isPanning ? 'cursor-grabbing' : 'cursor-grab'),
        )}
      >
        <img
          src={image.objectUrl}
          alt={t('surface.uploadedImage', { name: metadata.name })}
          width={metadata.width}
          height={metadata.height}
          draggable={false}
          className="absolute top-0 left-0 max-w-none origin-top-left select-none"
          style={{
            transform: `translate(${String(transform.tx)}px, ${String(transform.ty)}px) scale(${String(transform.scale)})`,
            // Above 100% the user is inspecting pixels, so show them rather
            // than a smoothed guess at what lies between them.
            imageRendering: transform.scale > 1 ? 'pixelated' : 'auto',
          }}
        />
      </div>
    </div>
  )
}
