import { ImageUp, Loader2 } from 'lucide-react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import {
  ACCEPTED_EXTENSIONS,
  ACCEPTED_MIME_TYPES,
  DEFAULT_UPLOAD_LIMITS,
} from '@/config/limits'
import { cn } from '@/lib/cn'
import { formatBytes } from '@/lib/format'

/**
 * Drag-and-drop upload target.
 *
 * Built around a real `<label>` wrapping a visually-hidden `<input type=file>`:
 * clicking anywhere in the zone opens the picker natively, the input is in the
 * tab order, and Enter/Space open the picker — all without JavaScript, and
 * without inventing ARIA for a control the platform already provides.
 */
export function Dropzone({
  onFileSelected,
  isLoading = false,
  compact = false,
  className,
}: {
  onFileSelected: (file: File) => void
  isLoading?: boolean
  /** Denser presentation for when an image is already loaded. */
  compact?: boolean
  className?: string
}) {
  const { t, i18n } = useTranslation('upload')
  const [isDraggingOver, setIsDraggingOver] = useState(false)
  // dragenter/dragleave fire for every descendant, so a boolean alone flickers
  // as the pointer crosses child elements. Counting enters and leaves is the
  // standard fix.
  const dragDepth = useRef(0)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0]
      if (file !== undefined) onFileSelected(file)
    },
    [onFileSelected],
  )

  // Pasting an image is the fastest path from a screenshot to a result, and
  // costs one listener.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const file = event.clipboardData?.files[0]
      if (file !== undefined) {
        event.preventDefault()
        onFileSelected(file)
      }
    }
    window.addEventListener('paste', onPaste)
    return () => { window.removeEventListener('paste', onPaste) }
  }, [onFileSelected])

  const maxSize = formatBytes(DEFAULT_UPLOAD_LIMITS.maxFileSizeBytes, 0, i18n.language)

  return (
    <label
      data-testid="dropzone"
      onDragEnter={(event) => {
        event.preventDefault()
        dragDepth.current += 1
        setIsDraggingOver(true)
      }}
      onDragOver={(event) => {
        // Required, or the browser opens the file instead of firing onDrop.
        event.preventDefault()
      }}
      onDragLeave={(event) => {
        event.preventDefault()
        dragDepth.current -= 1
        if (dragDepth.current <= 0) {
          dragDepth.current = 0
          setIsDraggingOver(false)
        }
      }}
      onDrop={(event) => {
        event.preventDefault()
        dragDepth.current = 0
        setIsDraggingOver(false)
        handleFiles(event.dataTransfer.files)
      }}
      className={cn(
        'group relative flex cursor-pointer flex-col items-center justify-center',
        'rounded-lg border border-dashed border-border bg-surface/40 text-center',
        'transition-colors duration-150',
        compact ? 'gap-2 px-4 py-6' : 'gap-3 px-6 py-14',
        isDraggingOver
          ? 'border-accent bg-accent/8'
          : 'hover:border-muted-foreground/50 hover:bg-surface/70',
        isLoading && 'pointer-events-none opacity-70',
        'has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2',
        'has-[:focus-visible]:outline-[var(--color-ring)]',
        className,
      )}
    >
      <input
        ref={inputRef}
        type="file"
        // Advisory only: the file's contents are sniffed after selection, so a
        // renamed file is rejected regardless of what this filter allows.
        accept={ACCEPTED_MIME_TYPES}
        className="sr-only"
        disabled={isLoading}
        onChange={(event) => {
          handleFiles(event.target.files)
          // Reset so selecting the same file twice fires change again.
          event.target.value = ''
        }}
      />

      {isLoading ? (
        <Loader2 aria-hidden="true" className="size-6 animate-spin text-muted-foreground" />
      ) : (
        <ImageUp
          aria-hidden="true"
          className={cn(
            'size-6 transition-colors',
            isDraggingOver ? 'text-accent' : 'text-muted-foreground',
          )}
        />
      )}

      <div>
        <p className={cn('font-medium text-foreground', compact ? 'text-sm' : 'text-base')}>
          {isLoading ? t('dropzone.loading') : t('dropzone.idle')}
        </p>
        {!isLoading && (
          <p className="mt-0.5 text-sm text-muted-foreground">{t('dropzone.browse')}</p>
        )}
      </div>

      {!compact && !isLoading && (
        <p className="text-xs text-muted-foreground">
          {/* Extensions and byte units are the same in every language; only the
              word joining them changes. */}
          {t('dropzone.limits', { formats: ACCEPTED_EXTENSIONS.join(' · '), maxSize })}
        </p>
      )}
    </label>
  )
}
