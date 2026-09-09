import { beforeEach, describe, expect, it } from 'vitest'
import {
  DEFAULT_DENOISE,
  DEFAULT_QUALITY,
  DEFAULT_SCALE,
  isLossy,
  nearestSupportedScale,
  preferredModel,
  useEnhancementStore,
} from './useEnhancementStore'
import type { ModelInfo } from '@/types/system'

function model(overrides: Partial<ModelInfo> & { id: string }): ModelInfo {
  return {
    name: overrides.id,
    description: '',
    arch: 'RRDBNet',
    scale: 4,
    supportsDenoise: false,
    supportedScales: [4, 8],
    downloaded: true,
    sizeMb: 10,
    ...overrides,
  }
}

beforeEach(() => {
  // Values only: the actions on the store are stable and must survive a reset.
  useEnhancementStore.setState({
    modelId: null,
    scale: DEFAULT_SCALE,
    denoiseStrength: DEFAULT_DENOISE,
    sharpenStrength: 0,
    format: 'png',
    quality: DEFAULT_QUALITY,
    preserveMetadata: true,
    activeJobId: null,
    modelChosen: false,
    denoiseChosen: false,
  })
})

// ------------------------------------------- unset versus a value that looks set
//
// Phase 4 F1 measured Enhancement Mode producing byte-identical output to
// Standard, because the client could not tell "the user picked this model" from
// "this is the model we adopted on load", nor "the user set denoise to 1.0"
// from "1.0 is the slider's starting value". The store now records the
// difference, and these lock it down.

describe('explicit choice tracking', () => {
  it('starts with nothing chosen, so the backend decides the mode defaults', () => {
    expect(useEnhancementStore.getState().modelChosen).toBe(false)
    expect(useEnhancementStore.getState().denoiseChosen).toBe(false)
  })

  it('adopting a default model is not the user choosing one', () => {
    useEnhancementStore.getState().adoptDefaults([model({ id: 'a' })])

    expect(useEnhancementStore.getState().modelId).toBe('a')
    expect(useEnhancementStore.getState().modelChosen).toBe(false)
  })

  it('picking a model from the dropdown is a choice', () => {
    useEnhancementStore.getState().setModel('b', [4])

    expect(useEnhancementStore.getState().modelChosen).toBe(true)
  })

  it('moving the denoise slider is a choice', () => {
    useEnhancementStore.getState().setDenoiseStrength(0.4)

    expect(useEnhancementStore.getState().denoiseChosen).toBe(true)
  })

  it('treats an explicit zero as a choice, not as absence', () => {
    // The whole bug was a value being read as "unset" because it equalled the
    // default. 0.0 is a real denoise setting - fully the wdn weights.
    useEnhancementStore.getState().setDenoiseStrength(0)

    expect(useEnhancementStore.getState().denoiseStrength).toBe(0)
    expect(useEnhancementStore.getState().denoiseChosen).toBe(true)
  })

  it('treats setting the slider to its own default as a choice', () => {
    useEnhancementStore.getState().setDenoiseStrength(DEFAULT_DENOISE)

    expect(useEnhancementStore.getState().denoiseChosen).toBe(true)
  })

  it('changing mode is not a model or denoise choice', () => {
    // setMode must stay a one-field action. If it started applying the mode's
    // model and denoise it would duplicate mode_planner in the client, which
    // is what F1 recommended against.
    useEnhancementStore.getState().adoptDefaults([model({ id: 'a' })])
    useEnhancementStore.getState().setMode('creative')

    expect(useEnhancementStore.getState().mode).toBe('creative')
    expect(useEnhancementStore.getState().modelId).toBe('a')
    expect(useEnhancementStore.getState().modelChosen).toBe(false)
    expect(useEnhancementStore.getState().denoiseChosen).toBe(false)
  })
})

// ------------------------------------------------------------ scale validity

describe('nearestSupportedScale', () => {
  it('keeps a factor the model can already produce', () => {
    expect(nearestSupportedScale(4, [4, 8])).toBe(4)
  })

  it('moves to the closest factor rather than always the first', () => {
    // 8x on a model that only does 2x has one honest answer.
    expect(nearestSupportedScale(8, [2])).toBe(2)
    expect(nearestSupportedScale(2, [4, 8])).toBe(4)
  })

  it('leaves the value alone when nothing is known yet', () => {
    expect(nearestSupportedScale(4, [])).toBe(4)
  })
})

describe('selecting a model', () => {
  it('coerces a scale the new model cannot produce', () => {
    useEnhancementStore.getState().setScale(8)

    useEnhancementStore.getState().setModel('RealESRGAN_x2plus', [2])

    expect(useEnhancementStore.getState().scale).toBe(2)
    expect(useEnhancementStore.getState().modelId).toBe('RealESRGAN_x2plus')
  })

  it('leaves a still-valid scale untouched', () => {
    useEnhancementStore.getState().setScale(8)

    useEnhancementStore.getState().setModel('RealESRGAN_x4plus', [4, 8])

    expect(useEnhancementStore.getState().scale).toBe(8)
  })
})

// ---------------------------------------------------------------- defaults

describe('preferredModel', () => {
  it('prefers a model whose weights are on disk', () => {
    const chosen = preferredModel([
      model({ id: 'missing', downloaded: false }),
      model({ id: 'ready', downloaded: true }),
    ])

    expect(chosen?.id).toBe('ready')
  })

  it('falls back to the first entry when none are downloaded', () => {
    const chosen = preferredModel([
      model({ id: 'first', downloaded: false }),
      model({ id: 'second', downloaded: false }),
    ])

    expect(chosen?.id).toBe('first')
  })

  it('has nothing to prefer in an empty registry', () => {
    expect(preferredModel([])).toBeUndefined()
  })
})

describe('adoptDefaults', () => {
  it('selects a model and a scale it supports', () => {
    useEnhancementStore.getState().setScale(8)

    useEnhancementStore
      .getState()
      .adoptDefaults([model({ id: 'two-x', scale: 2, supportedScales: [2] })])

    expect(useEnhancementStore.getState().modelId).toBe('two-x')
    expect(useEnhancementStore.getState().scale).toBe(2)
  })

  it('does not override a choice the user has already made', () => {
    useEnhancementStore.getState().setModel('chosen', [4, 8])

    useEnhancementStore.getState().adoptDefaults([model({ id: 'other' })])

    expect(useEnhancementStore.getState().modelId).toBe('chosen')
  })

  it('waits when the registry is empty', () => {
    useEnhancementStore.getState().adoptDefaults([])

    expect(useEnhancementStore.getState().modelId).toBeNull()
  })
})

// ------------------------------------------------------------------ formats

describe('isLossy', () => {
  it('knows quality is meaningless for PNG', () => {
    expect(isLossy('png')).toBe(false)
    expect(isLossy('jpeg')).toBe(true)
    expect(isLossy('webp')).toBe(true)
  })
})

describe('defaults', () => {
  it('starts on a lossless format so an upscale is not immediately recompressed', () => {
    expect(useEnhancementStore.getState().format).toBe('png')
  })

  it('starts with no sharpening, since it is a post-process the user must ask for', () => {
    expect(useEnhancementStore.getState().sharpenStrength).toBe(0)
  })

  it('starts with the model as shipped rather than a blend', () => {
    // 1.0 is the standard checkpoint untouched, which is what omitting the
    // setting would give.
    expect(useEnhancementStore.getState().denoiseStrength).toBe(1)
  })
})
