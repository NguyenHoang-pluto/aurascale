import type { LucideIcon } from 'lucide-react'

interface PlaceholderPageProps {
  icon: LucideIcon
  title: string
  description: string
  /** Which build phase delivers this screen — kept visible so the scaffold is never mistaken for a finished feature. */
  phase: string
}

/**
 * Honest scaffold placeholder. Phase 2 ships structure and tooling only;
 * each screen states which phase implements it rather than showing fake UI.
 */
export function PlaceholderPage({ icon: Icon, title, description, phase }: PlaceholderPageProps) {
  return (
    <div className="flex flex-1 items-center justify-center p-6">
      <div className="max-w-md text-center">
        <div className="mx-auto grid size-11 place-items-center rounded-lg border border-border bg-surface">
          <Icon className="size-5 text-muted-foreground" aria-hidden="true" />
        </div>
        <h1 className="mt-4 text-lg font-semibold tracking-tight">{title}</h1>
        <p className="mt-2 text-sm text-muted-foreground">{description}</p>
        <p className="mt-4 inline-flex rounded-full border border-border bg-muted px-2.5 py-1 font-mono text-xs text-muted-foreground">
          {phase}
        </p>
      </div>
    </div>
  )
}
