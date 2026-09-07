import { ArrowRight, Download, ImageUp, SlidersHorizontal, Sparkles } from 'lucide-react'
import { PhaseNotice } from '@/components/feedback/PhaseNotice'
import { WorkspaceLayout } from '@/components/layout/WorkspaceLayout'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'

const FLOW_STEPS = [
  { icon: ImageUp, label: 'Upload' },
  { icon: SlidersHorizontal, label: 'Compare' },
  { icon: Sparkles, label: 'Enhance' },
  { icon: Download, label: 'Download' },
] as const

/**
 * The four-step flow, stated plainly at the top of the workspace (§ 31).
 * It is a static explanation of what this screen does, not a progress tracker —
 * it will be replaced by live step state once jobs exist in Phase 7.
 */
function FlowSummary() {
  return (
    <ol className="flex flex-wrap items-center gap-x-1 gap-y-2 text-xs text-muted-foreground">
      {FLOW_STEPS.map((step, index) => (
        <li key={step.label} className="flex items-center gap-1">
          <span className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1">
            <step.icon aria-hidden="true" className="size-3.5" />
            {step.label}
          </span>
          {index < FLOW_STEPS.length - 1 && (
            <ArrowRight aria-hidden="true" className="size-3 opacity-50" />
          )}
        </li>
      ))}
    </ol>
  )
}

export function EnhancePage() {
  return (
    <WorkspaceLayout
      toolbar={<FlowSummary />}
      canvas={
        <div className="flex min-h-0 flex-1 items-center justify-center overflow-auto p-6">
          <PhaseNotice
            title="Image workspace"
            description="The drag-and-drop upload area and the image viewer live here. The before/after comparison replaces this region once a job has produced a result."
            phase="Phase 4"
            className="w-full max-w-lg bg-surface/40"
          />
        </div>
      }
      rail={
        <>
          <Panel>
            <PanelHeader>
              <PanelTitle>Enhancement</PanelTitle>
            </PanelHeader>
            <PanelContent>
              <PhaseNotice
                title="Model and scale"
                description="Model selection, upscale factor and the enhancement options are wired to the backend here."
                phase="Phase 8"
                className="border-0 px-0 py-4"
              />
            </PanelContent>
          </Panel>

          <Panel>
            <PanelHeader>
              <PanelTitle>Output</PanelTitle>
            </PanelHeader>
            <PanelContent>
              <PhaseNotice
                title="Format and quality"
                description="Output format, quality and metadata handling."
                phase="Phase 8"
                className="border-0 px-0 py-4"
              />
            </PanelContent>
          </Panel>
        </>
      }
    />
  )
}
