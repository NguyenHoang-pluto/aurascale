import { create } from 'zustand'
import type { ModelInfo } from '@/types/system'
import type { EnhancementMode, OutputFormat, TargetResolution } from '@/types/job'

/**
 * The enhancement and output settings, plus the job they were submitted as.
 *
 * Client state, so it lives here rather than in TanStack Query: nothing on the
 * server knows what the user has selected until a job is submitted.
 *
 * The one rule this store enforces is that `scale` is always a factor the
 * selected model can actually produce. It does not decide which factors those
 * are — `/api/models` publishes `supportedScales` for each model, derived from
 * the same planner that validates a job, so the UI cannot drift from what the
 * backend will accept.
 */

export const DEFAULT_SCALE = 4
/** Fidelity first, and what every existing job already did. */
export const DEFAULT_MODE: EnhancementMode = 'standard'

/**
 * How the output size is expressed: as a factor, or as a destination.
 *
 * Kept as an explicit union rather than "target is null means scale", so the
 * two are visibly different choices in the code as well as in the UI.
 */
export type OutputSizing = { kind: 'scale' } | { kind: 'target'; target: TargetResolution }

export const DEFAULT_SIZING: OutputSizing = { kind: 'scale' }
export const DEFAULT_QUALITY = 92
/** 1.0 is the model's own weights, which denoise the most (see the panel copy). */
export const DEFAULT_DENOISE = 1
export const QUALITY_RANGE = { min: 50, max: 100 } as const

/** Formats where `quality` means something. PNG is lossless. */
export const LOSSY_FORMATS: readonly OutputFormat[] = ['jpeg', 'webp']

export function isLossy(format: OutputFormat): boolean {
  return LOSSY_FORMATS.includes(format)
}

interface EnhancementState {
  modelId: string | null
  /**
   * Whether `modelId` is a choice the user made, or just what was adopted when
   * the registry loaded.
   *
   * The distinction is the whole of the Phase 4 F1 fix. `mode_planner` lets an
   * explicit model win over the mode's own - deliberately, so clients written
   * before modes keep working - but the client used to send one on every
   * request, so the mode's model could never apply and Creative produced
   * byte-identical output to Standard. A value equal to the default is not the
   * same as a value nobody set, and the request now says which it is.
   */
  modelChosen: boolean
  mode: EnhancementMode
  sizing: OutputSizing
  scale: number
  /** 0-1. Only meaningful for models with `supportsDenoise`. */
  denoiseStrength: number
  /** The same distinction for denoise. See `modelChosen`. */
  denoiseChosen: boolean
  /** 0-1. A post-process, never presented as an AI feature. */
  sharpenStrength: number
  format: OutputFormat
  quality: number
  preserveMetadata: boolean

  /** The job currently being watched, if any. */
  activeJobId: string | null

  setMode: (mode: EnhancementMode) => void
  setSizing: (sizing: OutputSizing) => void
  setModel: (modelId: string, supportedScales: readonly number[]) => void
  setScale: (scale: number) => void
  setDenoiseStrength: (value: number) => void
  setSharpenStrength: (value: number) => void
  setFormat: (format: OutputFormat) => void
  setQuality: (quality: number) => void
  setPreserveMetadata: (preserve: boolean) => void

  /** Adopt a default model once the registry has loaded. */
  adoptDefaults: (models: readonly ModelInfo[]) => void

  setActiveJob: (jobId: string | null) => void
}

/**
 * The closest usable factor when the current one is not on offer.
 *
 * Prefers the nearest supported factor rather than always snapping to the
 * first: a user on 8x switching to a 2x-only model should land on 2x, and a
 * user on 2x switching to a 4x model should land on 4x, without either feeling
 * arbitrary.
 */
export function nearestSupportedScale(scale: number, supported: readonly number[]): number {
  if (supported.length === 0) return scale
  if (supported.includes(scale)) return scale

  return supported.reduce((best, candidate) =>
    Math.abs(candidate - scale) < Math.abs(best - scale) ? candidate : best,
  )
}

/** The model a fresh session should start on: the first one ready to run. */
export function preferredModel(models: readonly ModelInfo[]): ModelInfo | undefined {
  return models.find((model) => model.downloaded) ?? models[0]
}

export const useEnhancementStore = create<EnhancementState>()((set, get) => ({
  modelId: null,
  modelChosen: false,
  denoiseChosen: false,
  mode: DEFAULT_MODE,
  sizing: DEFAULT_SIZING,
  scale: DEFAULT_SCALE,
  denoiseStrength: DEFAULT_DENOISE,
  sharpenStrength: 0,
  format: 'png',
  quality: DEFAULT_QUALITY,
  preserveMetadata: true,
  activeJobId: null,

  setModel: (modelId, supportedScales) => {
    // Only reached from the model dropdown, which is a user action.
    // `adoptDefaults` writes `modelId` directly and leaves `modelChosen` alone.
    set({ modelId, modelChosen: true, scale: nearestSupportedScale(get().scale, supportedScales) })
  },

  setMode: (mode) => { set({ mode }) },
  setSizing: (sizing) => { set({ sizing }) },
  setScale: (scale) => { set({ scale }) },
  setDenoiseStrength: (denoiseStrength) => { set({ denoiseStrength, denoiseChosen: true }) },
  setSharpenStrength: (sharpenStrength) => { set({ sharpenStrength }) },
  setFormat: (format) => { set({ format }) },
  setQuality: (quality) => { set({ quality }) },
  setPreserveMetadata: (preserveMetadata) => { set({ preserveMetadata }) },

  adoptDefaults: (models) => {
    if (get().modelId !== null || models.length === 0) return

    const model = preferredModel(models)
    if (model === undefined) return

    set({
      modelId: model.id,
      scale: nearestSupportedScale(get().scale, model.supportedScales),
    })
  },

  setActiveJob: (activeJobId) => { set({ activeJobId }) },
}))
