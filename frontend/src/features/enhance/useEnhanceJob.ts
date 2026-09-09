import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'
import { toApiError } from '@/services/apiClient'
import { cancelJob, createJob } from '@/services/jobsApi'
import { useComparisonStore } from '@/stores/useComparisonStore'
import { useEnhancementStore, isLossy } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import type { ProblemDetail } from '@/types/api'
import type { CreateJobRequest, EnhanceSettings, JobRecord } from '@/types/job'
import { JOB_QUERY_KEY } from './useJobProgress'

/**
 * Submitting the loaded image, and stopping it again.
 *
 * The request is assembled from the settings store and the loaded file, with
 * one rule: an option the backend would reject for this model is not sent at
 * all. Sending `denoiseStrength` to a model with no denoise pair is a 422, so
 * the control that produced it is disabled and the value omitted.
 */

export interface EnhanceJobResult {
  submit: () => void
  cancel: () => void
  isSubmitting: boolean
  isCancelling: boolean
  problem: ProblemDetail | undefined
  reset: () => void
}

export function buildSettings(options: {
  sharpenStrength: number
  denoiseStrength: number
  supportsDenoise: boolean
  /** Whether the user moved the slider, as opposed to it holding its default. */
  denoiseChosen: boolean
}): EnhanceSettings {
  const settings: EnhanceSettings = {}

  if (options.sharpenStrength > 0) settings.sharpenStrength = options.sharpenStrength
  // Two separate reasons to omit it, and they are not the same reason.
  //
  //   * the model has no paired weights, so the backend refuses the field
  //     rather than ignoring it, and rightly so;
  //   * the user has not touched the slider, so there is nothing to say. Sending
  //     the default anyway is what made Enhancement Mode inert - `resolve_denoise`
  //     lets an explicit value win, and an untouched slider was still explicit.
  //
  // Note this is `denoiseChosen`, not `denoiseStrength > 0`: zero is a real
  // setting (fully the wdn weights), not an absence.
  if (options.supportsDenoise && options.denoiseChosen) {
    settings.denoiseStrength = options.denoiseStrength
  }

  return settings
}

export function useEnhanceJob(options: { supportsDenoise: boolean }): EnhanceJobResult {
  const queryClient = useQueryClient()
  const source = useWorkspaceStore((s) => s.source)

  const modelId = useEnhancementStore((s) => s.modelId)
  const modelChosen = useEnhancementStore((s) => s.modelChosen)
  const denoiseChosen = useEnhancementStore((s) => s.denoiseChosen)
  const mode = useEnhancementStore((s) => s.mode)
  const sizing = useEnhancementStore((s) => s.sizing)
  const scale = useEnhancementStore((s) => s.scale)
  const format = useEnhancementStore((s) => s.format)
  const quality = useEnhancementStore((s) => s.quality)
  const preserveMetadata = useEnhancementStore((s) => s.preserveMetadata)
  const sharpenStrength = useEnhancementStore((s) => s.sharpenStrength)
  const denoiseStrength = useEnhancementStore((s) => s.denoiseStrength)
  const activeJobId = useEnhancementStore((s) => s.activeJobId)
  const setActiveJob = useEnhancementStore((s) => s.setActiveJob)
  const captureBefore = useComparisonStore((s) => s.captureBefore)
  const clearBefore = useComparisonStore((s) => s.clearBefore)

  const submission = useMutation({
    mutationFn: () => {
      if (source === null || modelId === null) {
        throw new Error('nothing to submit')
      }

      const request: CreateJobRequest = {
        file: source.file,
        // Sent only when the user chose it. Left out otherwise so `mode` can
        // supply the model, which is what `mode_planner` is for - the planner
        // stays the single source of truth and the client never learns which
        // model a mode implies.
        ...(modelChosen ? { model: modelId } : {}),
        scale,
        format,
        preserveMetadata,
        settings: buildSettings({
          sharpenStrength,
          denoiseStrength,
          supportsDenoise: options.supportsDenoise,
          denoiseChosen,
        }),
        // Quality is meaningless for a lossless format, so it is not sent.
        ...(isLossy(format) ? { quality } : {}),
        mode,
        // A target replaces the factor rather than accompanying it; the
        // serializer sends whichever one is set.
        ...(sizing.kind === 'target' ? { target: sizing.target } : {}),
      }

      return createJob(request)
    },
    onSuccess: (created) => {
      // Snapshot what was actually submitted, with its own object URL. Reading
      // the workspace later would compare against whatever image happens to be
      // loaded then, which need not be the one this job ran on.
      if (source !== null) {
        captureBefore(source.file, {
          width: source.metadata.width,
          height: source.metadata.height,
        })
      }
      setActiveJob(created.jobId)
      // Seed the record so the panel has a status immediately instead of a
      // blank frame while the first poll lands.
      queryClient.setQueryData<Partial<JobRecord>>(JOB_QUERY_KEY(created.jobId), undefined)
    },
  })

  const cancellation = useMutation({
    mutationFn: () => {
      if (activeJobId === null) throw new Error('no active job')
      return cancelJob(activeJobId)
    },
    onSuccess: () => {
      if (activeJobId !== null) {
        void queryClient.invalidateQueries({ queryKey: JOB_QUERY_KEY(activeJobId) })
      }
    },
  })

  const reset = useCallback(() => {
    setActiveJob(null)
    clearBefore()
    submission.reset()
    cancellation.reset()
  }, [cancellation, clearBefore, setActiveJob, submission])

  const failure = submission.error ?? cancellation.error

  return {
    submit: () => { submission.mutate() },
    cancel: () => { cancellation.mutate() },
    isSubmitting: submission.isPending,
    isCancelling: cancellation.isPending,
    problem: failure != null ? toApiError(failure).problem : undefined,
    reset,
  }
}
