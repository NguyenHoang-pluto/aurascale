import { useId, type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { cn } from '@/lib/cn'
import { Badge } from './badge'
import { Label } from './label'

/**
 * A labelled settings row.
 *
 * Wires `htmlFor`/`id` and `aria-describedby` through a render prop, so every
 * control in the settings panel is labelled and described without each caller
 * inventing its own ids.
 *
 * `comingSoon` renders the honest "Coming soon" marker required by § 6 — a
 * control that is not implemented yet is visibly disabled and says so, rather
 * than being wired to nothing.
 */
export function Field({
  label,
  description,
  comingSoon = false,
  orientation = 'vertical',
  className,
  children,
}: {
  label: string
  description?: string
  comingSoon?: boolean
  orientation?: 'vertical' | 'horizontal'
  className?: string
  children: (ids: { id: string; describedBy: string | undefined }) => ReactNode
}) {
  const { t } = useTranslation('common')
  const id = useId()
  const descriptionId = `${id}-description`
  const describedBy = description !== undefined ? descriptionId : undefined

  return (
    <div
      className={cn(
        orientation === 'vertical'
          ? 'flex flex-col gap-2'
          : 'flex items-center justify-between gap-4',
        comingSoon && 'opacity-60',
        className,
      )}
    >
      <div className={cn('flex flex-col gap-0.5', orientation === 'horizontal' && 'min-w-0')}>
        <div className="flex items-center gap-2">
          <Label htmlFor={id}>{label}</Label>
          {comingSoon && (
            <Badge tone="neutral" className="font-normal">
              {t('state.comingSoon')}
            </Badge>
          )}
        </div>
        {description !== undefined && (
          <p id={descriptionId} className="text-xs text-muted-foreground">
            {description}
          </p>
        )}
      </div>
      {children({ id, describedBy })}
    </div>
  )
}
