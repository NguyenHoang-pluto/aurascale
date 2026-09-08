import * as Collapsible from '@radix-ui/react-collapsible'
import { AlertCircle, ChevronRight, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/cn'
import { Button } from '@/components/ui/button'

/**
 * The single way errors are shown to the user (§ 10).
 *
 * `title` and `detail` are human sentences. `technical` — a traceback, a CUDA
 * message, a status line — is hidden behind a collapsed disclosure so it is
 * available to a developer without being the first thing a user reads.
 */
export function ErrorPanel({
  title,
  detail,
  technical,
  code,
  onRetry,
  retryLabel,
  className,
}: {
  title: string
  detail: string
  technical?: string
  /** Stable error code, shown alongside the technical detail for bug reports. */
  code?: string
  onRetry?: () => void
  /** Overrides the default "Try again"; already-translated text. */
  retryLabel?: string
  className?: string
}) {
  const { t } = useTranslation('errors')
  const [isOpen, setIsOpen] = useState(false)
  const hasTechnical = technical !== undefined || code !== undefined

  return (
    <div
      role="alert"
      className={cn(
        'rounded-lg border border-destructive/40 bg-destructive/8 p-4',
        className,
      )}
    >
      <div className="flex gap-3">
        <AlertCircle
          aria-hidden="true"
          className="mt-0.5 size-4 shrink-0 text-destructive"
        />
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-semibold text-foreground">{title}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{detail}</p>

          {hasTechnical && (
            <Collapsible.Root open={isOpen} onOpenChange={setIsOpen} className="mt-3">
              <Collapsible.Trigger asChild>
                <button
                  type="button"
                  className={cn(
                    'inline-flex items-center gap-1 rounded text-xs font-medium',
                    'text-muted-foreground transition-colors hover:text-foreground',
                  )}
                >
                  <ChevronRight
                    aria-hidden="true"
                    className={cn(
                      'size-3 transition-transform duration-150',
                      isOpen && 'rotate-90',
                    )}
                  />
                  {t('technicalDetails')}
                </button>
              </Collapsible.Trigger>
              <Collapsible.Content>
                <div className="mt-2 rounded border border-border bg-background p-2.5">
                  {code !== undefined && (
                    <p className="font-mono text-xs text-muted-foreground">code: {code}</p>
                  )}
                  {technical !== undefined && (
                    <pre className="mt-1 overflow-x-auto font-mono text-xs whitespace-pre-wrap text-muted-foreground">
                      {technical}
                    </pre>
                  )}
                </div>
              </Collapsible.Content>
            </Collapsible.Root>
          )}

          {onRetry !== undefined && (
            <Button variant="outline" size="sm" onClick={onRetry} className="mt-3">
              <RotateCcw aria-hidden="true" />
              {retryLabel ?? t('retry')}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
