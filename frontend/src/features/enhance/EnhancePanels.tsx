import { useEffect } from 'react'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { useModels } from '@/features/system/useSystemInfo'
import { useEnhancementStore } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import { EnhancementControls } from './EnhancementControls'
import { describeBlocker } from './jobPresentation'
import { JobPanel } from './JobPanel'
import { OutputControls } from './OutputControls'
import { useEnhanceJob } from './useEnhanceJob'
import { useJobProgress } from './useJobProgress'

/**
 * The settings rail: what to run, what to write, and the job itself.
 *
 * Composed here rather than in the page so the page stays a layout, and so the
 * three panels share one reading of the model registry — they all depend on
 * the same capability data and must not disagree about it.
 */
export function EnhancePanels() {
  const source = useWorkspaceStore((s) => s.source)
  const models = useModels()

  const modelId = useEnhancementStore((s) => s.modelId)
  const adoptDefaults = useEnhancementStore((s) => s.adoptDefaults)
  const activeJobId = useEnhancementStore((s) => s.activeJobId)

  // Pick a starting model once the registry is known. The backend has its own
  // default, but the panel has to show *something* selected, and showing a
  // guess that the server might not share would be worse than asking it.
  useEffect(() => {
    if (models.data !== undefined) adoptDefaults(models.data)
  }, [models.data, adoptDefaults])

  const selected = models.data?.find((model) => model.id === modelId)
  const supportsDenoise = selected?.supportsDenoise ?? false

  const { job, live } = useJobProgress(activeJobId)
  const enhancement = useEnhanceJob({ supportsDenoise })

  const isRunning = job !== undefined && (job.status === 'queued' || job.status === 'processing')

  return (
    <>
      <Panel>
        <PanelHeader>
          <PanelTitle>Enhancement</PanelTitle>
        </PanelHeader>
        <PanelContent>
          <fieldset disabled={isRunning} className="contents">
            <EnhancementControls
              models={models.data ?? []}
              isLoading={models.isPending}
              selected={selected}
              {...(models.isError
                ? {
                    problem: {
                      title: 'Cannot load the model list',
                      detail:
                        models.reason ??
                        'The backend did not return the available models. Check that it is running.',
                    },
                  }
                : {})}
            />
          </fieldset>
        </PanelContent>
      </Panel>

      <Panel>
        <PanelHeader>
          <PanelTitle>Output</PanelTitle>
        </PanelHeader>
        <PanelContent>
          <fieldset disabled={isRunning} className="contents">
            <OutputControls source={source?.metadata} />
          </fieldset>
        </PanelContent>
      </Panel>

      <Panel>
        <PanelHeader>
          <PanelTitle>Job</PanelTitle>
        </PanelHeader>
        <PanelContent>
          <JobPanel
            job={job}
            live={live}
            canSubmit={source !== null && selected !== undefined && selected.downloaded}
            disabledReason={describeBlocker(source !== null, selected)}
            isSubmitting={enhancement.isSubmitting}
            isCancelling={enhancement.isCancelling}
            problem={enhancement.problem}
            onSubmit={enhancement.submit}
            onCancel={enhancement.cancel}
            onReset={enhancement.reset}
          />
        </PanelContent>
      </Panel>
    </>
  )
}
