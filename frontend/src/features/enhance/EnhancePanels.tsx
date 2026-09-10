import { useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { useModels } from '@/features/system/useSystemInfo'
import { useEnhancementStore } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import { MODE_DEFAULT_MODEL } from '@/types/job'
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
  const { t } = useTranslation(['enhance', 'job'])
  const source = useWorkspaceStore((s) => s.source)
  const models = useModels()

  const modelId = useEnhancementStore((s) => s.modelId)
  const modelChosen = useEnhancementStore((s) => s.modelChosen)
  const mode = useEnhancementStore((s) => s.mode)
  const adoptDefaults = useEnhancementStore((s) => s.adoptDefaults)
  const activeJobId = useEnhancementStore((s) => s.activeJobId)

  // Pick a starting model once the registry is known. The backend has its own
  // default, but the panel has to show *something* selected, and showing a
  // guess that the server might not share would be worse than asking it.
  useEffect(() => {
    if (models.data !== undefined) adoptDefaults(models.data)
  }, [models.data, adoptDefaults])

  // What the panel should describe, which is not always what the store holds.
  //
  // `adoptDefaults` writes a `modelId` so the dropdown has something selected,
  // and leaves `modelChosen` false so `buildSettings` omits the field and the
  // backend's planner decides - that is the F1 behaviour and it stays. But the
  // adopted model is the first downloaded one, not the mode's, so reading
  // capabilities off it made Creative show `RealESRGAN_x4plus` and disable a
  // denoise control that the model actually running supports.
  //
  // Until the user chooses, the mode owns the answer. After that the choice
  // does, in every mode - which is the same precedence the backend applies.
  const effectiveModelId = modelChosen ? modelId : MODE_DEFAULT_MODEL[mode]
  const selected =
    models.data?.find((model) => model.id === effectiveModelId) ??
    // A build whose registry has no entry for the mode's model falls back to
    // the adopted one rather than rendering an empty panel.
    models.data?.find((model) => model.id === modelId)
  const supportsDenoise = selected?.supportsDenoise ?? false

  const { job, live } = useJobProgress(activeJobId)
  const enhancement = useEnhanceJob({ supportsDenoise })

  const isRunning = job !== undefined && (job.status === 'queued' || job.status === 'processing')

  return (
    <>
      <Panel>
        <PanelHeader>
          <PanelTitle>{t('enhance:panel.enhancement')}</PanelTitle>
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
                      title: t('enhance:model.loadFailed'),
                      detail: models.reason ?? t('enhance:model.loadFailedDetail'),
                    },
                  }
                : {})}
            />
          </fieldset>
        </PanelContent>
      </Panel>

      <Panel>
        <PanelHeader>
          <PanelTitle>{t('enhance:panel.output')}</PanelTitle>
        </PanelHeader>
        <PanelContent>
          <fieldset disabled={isRunning} className="contents">
            <OutputControls source={source?.metadata} />
          </fieldset>
        </PanelContent>
      </Panel>

      <Panel>
        <PanelHeader>
          <PanelTitle>{t('enhance:panel.job')}</PanelTitle>
        </PanelHeader>
        <PanelContent>
          <JobPanel
            job={job}
            live={live}
            canSubmit={source !== null && selected !== undefined && selected.downloaded}
            disabledReason={describeBlocker(t, source !== null, selected)}
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
