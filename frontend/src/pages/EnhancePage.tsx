import { ArrowRight, Download, ImageUp, SlidersHorizontal, Sparkles, X } from 'lucide-react'
import { useCallback } from 'react'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { WorkspaceLayout } from '@/components/layout/WorkspaceLayout'
import { Button } from '@/components/ui/button'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { EnhancePanels } from '@/features/enhance/EnhancePanels'
import { Dropzone } from '@/features/upload/Dropzone'
import { ImageInfoPanel } from '@/features/viewer/ImageInfoPanel'
import { ImageViewer } from '@/features/viewer/ImageViewer'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'

const FLOW_STEPS = [
  { icon: ImageUp, label: 'Upload' },
  { icon: SlidersHorizontal, label: 'Compare' },
  { icon: Sparkles, label: 'Enhance' },
  { icon: Download, label: 'Download' },
] as const

/**
 * The four-step flow, stated plainly at the top of the workspace (§ 31).
 * A static explanation of what this screen does; the live state of a running
 * job is reported by the Enhance panel, which has real measurements to show.
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
  const source = useWorkspaceStore((s) => s.source)
  const problem = useWorkspaceStore((s) => s.problem)
  const isLoading = useWorkspaceStore((s) => s.isLoading)
  const loadFile = useWorkspaceStore((s) => s.loadFile)
  const clear = useWorkspaceStore((s) => s.clear)
  const dismissProblem = useWorkspaceStore((s) => s.dismissProblem)

  const onFileSelected = useCallback(
    (file: File) => { void loadFile(file) },
    [loadFile],
  )

  return (
    <WorkspaceLayout
      toolbar={<FlowSummary />}
      canvas={
        source === null ? (
          <div className="flex min-h-0 flex-1 items-center justify-center overflow-y-auto p-6">
            <div className="flex w-full max-w-xl flex-col gap-4">
              <Dropzone onFileSelected={onFileSelected} isLoading={isLoading} />
              {problem !== null && (
                <ErrorPanel
                  title={problem.title}
                  detail={problem.detail}
                  code={problem.code}
                  {...(problem.technical !== undefined
                    ? { technical: problem.technical }
                    : {})}
                  onRetry={dismissProblem}
                  retryLabel="Dismiss"
                />
              )}
            </div>
          </div>
        ) : (
          <ImageViewer
            image={source}
            toolbarSlot={
              <div className="ml-auto flex items-center gap-2">
                <span
                  className="max-w-[16rem] truncate text-xs text-muted-foreground"
                  title={source.metadata.name}
                >
                  {source.metadata.name}
                </span>
                <Button variant="ghost" size="sm" onClick={clear}>
                  <X aria-hidden="true" />
                  Remove
                </Button>
              </div>
            }
          />
        )
      }
      rail={
        <>
          {source !== null && (
            <>
              <ImageInfoPanel metadata={source.metadata} />
              {problem !== null && (
                <ErrorPanel
                  title={problem.title}
                  detail={problem.detail}
                  code={problem.code}
                  {...(problem.technical !== undefined
                    ? { technical: problem.technical }
                    : {})}
                  onRetry={dismissProblem}
                  retryLabel="Dismiss"
                />
              )}
              <Panel>
                <PanelHeader>
                  <PanelTitle>Replace image</PanelTitle>
                </PanelHeader>
                <PanelContent>
                  <Dropzone compact onFileSelected={onFileSelected} isLoading={isLoading} />
                </PanelContent>
              </Panel>
            </>
          )}

          <EnhancePanels />
        </>
      }
    />
  )
}
