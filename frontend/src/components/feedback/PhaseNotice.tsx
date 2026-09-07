import { Construction } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/cn'

/**
 * Marks a region whose feature is not built yet.
 *
 * The project's rule is that nothing may look functional before it is. A region
 * that will hold a dropzone shows this notice rather than a dropzone that does
 * nothing when you drop on it.
 */
export function PhaseNotice({
  title,
  description,
  phase,
  className,
}: {
  title: string
  description: string
  /** e.g. "Phase 4". Shown verbatim. */
  phase: string
  className?: string
}) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3 rounded-lg border border-dashed',
        'border-border px-6 py-10 text-center',
        className,
      )}
    >
      <Construction aria-hidden="true" className="size-5 text-muted-foreground" />
      <div>
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        <p className="mx-auto mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>
      </div>
      <Badge tone="accent">Arrives in {phase}</Badge>
    </div>
  )
}
