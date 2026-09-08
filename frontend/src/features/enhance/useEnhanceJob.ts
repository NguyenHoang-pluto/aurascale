import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useCallback } from 'react'
import { toApiError } from '@/services/apiClient'
import { cancelJob, createJob } from '@/services/jobsApi'
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
}): EnhanceSettings {
  const settings: EnhanceSettings = {}

  if (options.sharpenStrength > 0) settings.sharpenStrength = options.sharpenStrength
  // Omitted entirely for a model without the paired weights: the backend
  // refuses the field rather than ignoring it, and rightly so.
  if (options.supportsDenoise) settings.denoiseStrength = options.denoiseStrength

  return settings
}

export function useEnhanceJob(options: { supportsDenoise: boolean }): EnhanceJobResult {
  const queryClient = useQueryClient()
  const source = useWorkspaceStore((s) => s.source)

  const modelId = useEnhancementStore((s) => s.modelId)
  const scale = useEnhancementStore((s) => s.scale)
  const format = useEnhancementStore((s) => s.format)
  const quality = useEnhancementStore((s) => s.quality)
  const preserveMetadata = useEnhancementStore((s) => s.preserveMetadata)
  const sharpenStrength = useEnhancementStore((s) => s.sharpenStrength)
  const denoiseStrength = useEnhancementStore((s) => s.denoiseStrength)
  const activeJobId = useEnhancementStore((s) => s.activeJobId)
  const setActiveJob = useEnhancementStore((s) => s.setActiveJob)

  const submission = useMutation({
    mutationFn: () => {
      if (source === null || modelId === null) {
        throw new Error('nothing to submit')
      }

      const request: CreateJobRequest = {
        file: source.file,
        model: modelId,
        scale,
        format,
        preserveMetadata,
        settings: buildSettings({
          sharpenStrength,
          denoiseStrength,
          supportsDenoise: options.supportsDenoise,
        }),
        // Quality is meaningless for a lossless format, so it is not sent.
        ...(isLossy(format) ? { quality } : {}),
      }

      return createJob(request)
    },
    onSuccess: (created) => {
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
    submission.reset()
    cancellation.reset()
  }, [cancellation, setActiveJob, submission])

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
