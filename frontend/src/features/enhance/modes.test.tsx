import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { EnhancePanels } from './EnhancePanels'
import { ModeControls } from './ModeControls'
import { createJob } from '@/services/jobsApi'
import { i18n } from '@/i18n'
import { renderWithProviders } from '@/test/renderWithProviders'
import { GPU_SYSTEM, HEALTH, stubApi } from '@/test/systemFixtures'
import {
  DEFAULT_MODE,
  DEFAULT_SCALE,
  DEFAULT_SIZING,
  useEnhancementStore,
} from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import { MODE_DEFAULT_MODEL, TARGET_LONG_EDGE } from '@/types/job'
import type { CreateJobRequest } from '@/types/job'
import type { ModelInfo } from '@/types/system'

/**
 * Mode and output-size selection.
 *
 * The controls express intent - what to prioritise, how big - and never a
 * model id. What reaches the wire is checked separately, because a selector
 * that looks right while serialising the wrong thing is the failure worth
 * catching.
 */

beforeEach(() => {
  useEnhancementStore.setState({
    modelId: 'RealESRGAN_x4plus',
    mode: DEFAULT_MODE,
    sizing: DEFAULT_SIZING,
    scale: DEFAULT_SCALE,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

// -------------------------------------------------------------- the selector

describe('the mode selector', () => {
  it('starts on Standard, which is what every existing job already did', () => {
    renderWithProviders(<ModeControls />)

    expect(screen.getByRole('radio', { name: 'Standard' })).toBeChecked()
    expect(useEnhancementStore.getState().mode).toBe('standard')
  })

  it('offers both modes and no model names', () => {
    renderWithProviders(<ModeControls />)

    expect(screen.getByRole('radio', { name: 'Standard' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: 'Creative' })).toBeInTheDocument()
    // "realesr-general-x4v3" is not something to ask a person to choose.
    expect(screen.queryByText(/realesr|RealESRGAN/)).not.toBeInTheDocument()
  })

  it('says what each mode is for', async () => {
    const user = userEvent.setup()
    renderWithProviders(<ModeControls />)

    expect(screen.getByText('Natural enhancement with high fidelity')).toBeInTheDocument()

    await user.click(screen.getByRole('radio', { name: 'Creative' }))

    expect(screen.getByText('Stronger detail and visual enhancement')).toBeInTheDocument()
  })

  it('records the choice', async () => {
    const user = userEvent.setup()
    renderWithProviders(<ModeControls />)

    await user.click(screen.getByRole('radio', { name: 'Creative' }))

    expect(useEnhancementStore.getState().mode).toBe('creative')
  })
})

describe('the output size selector', () => {
  it('starts on Scale, so existing behaviour is the default', () => {
    renderWithProviders(<ModeControls />)

    expect(screen.getByLabelText('Output size')).toHaveValue('scale')
    expect(useEnhancementStore.getState().sizing).toEqual({ kind: 'scale' })
  })

  it('offers every preset with its pixel count spelled out', () => {
    // "2K" means 2048 to a cinema and 2560 to a gamer; the number removes it.
    renderWithProviders(<ModeControls />)

    for (const [preset, pixels] of Object.entries(TARGET_LONG_EDGE)) {
      expect(
        screen.getByRole('option', {
          name: new RegExp(`${preset.toUpperCase()}.*${pixels} px`),
        }),
      ).toBeInTheDocument()
    }
  })

  it('switches to a target and back', async () => {
    const user = userEvent.setup()
    renderWithProviders(<ModeControls />)

    await user.selectOptions(screen.getByLabelText('Output size'), '4k')
    expect(useEnhancementStore.getState().sizing).toEqual({ kind: 'target', target: '4k' })

    await user.selectOptions(screen.getByLabelText('Output size'), 'scale')
    expect(useEnhancementStore.getState().sizing).toEqual({ kind: 'scale' })
  })

  it('explains that a target keeps the aspect ratio', async () => {
    const user = userEvent.setup()
    renderWithProviders(<ModeControls />)

    await user.selectOptions(screen.getByLabelText('Output size'), '8k')

    expect(screen.getByText(/aspect ratio is kept/)).toBeInTheDocument()
  })
})

// ------------------------------------------------------------ serialization

describe('what reaches the wire', () => {
  function capture() {
    const sent: FormData[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn((_url: string, init?: RequestInit) => {
        sent.push(init?.body as FormData)
        return Promise.resolve(
          new Response(JSON.stringify({ jobId: 'a'.repeat(32), status: 'queued', queuePosition: 0, createdAt: '2026-09-08T10:00:00Z' }), {
            headers: { 'content-type': 'application/json' },
          }),
        )
      }),
    )
    return sent
  }

  const base: CreateJobRequest = {
    file: new File([new Uint8Array(8)], 'x.png', { type: 'image/png' }),
    model: 'RealESRGAN_x4plus',
    scale: 4,
    format: 'png',
    preserveMetadata: true,
    settings: {},
  }

  it('sends a scale when no target is chosen', async () => {
    const sent = capture()

    await createJob(base)

    await waitFor(() => { expect(sent).toHaveLength(1) })
    expect(sent[0]!.get('scale')).toBe('4')
    expect(sent[0]!.get('target')).toBeNull()
  })

  it('sends a target instead of a scale, never both', async () => {
    // The backend refuses the pair, so sending both would turn a valid choice
    // into a validation error the user never made.
    const sent = capture()

    await createJob({ ...base, target: '4k' })

    await waitFor(() => { expect(sent).toHaveLength(1) })
    expect(sent[0]!.get('target')).toBe('4k')
    expect(sent[0]!.get('scale')).toBeNull()
  })

  it('sends the mode when one is set', async () => {
    const sent = capture()

    await createJob({ ...base, mode: 'creative' })

    await waitFor(() => { expect(sent).toHaveLength(1) })
    expect(sent[0]!.get('mode')).toBe('creative')
  })

  it('omits mode and target entirely for a request that sets neither', async () => {
    // The exact shape a client written before modes existed sends.
    const sent = capture()

    await createJob(base)

    await waitFor(() => { expect(sent).toHaveLength(1) })
    expect(sent[0]!.get('mode')).toBeNull()
    expect(sent[0]!.get('target')).toBeNull()
    expect(sent[0]!.get('model')).toBe('RealESRGAN_x4plus')
  })

  it('still sends the model alongside a mode, so an override survives', async () => {
    const sent = capture()

    await createJob({ ...base, mode: 'creative', model: 'RealESRGAN_x2plus' })

    await waitFor(() => { expect(sent).toHaveLength(1) })
    expect(sent[0]!.get('mode')).toBe('creative')
    expect(sent[0]!.get('model')).toBe('RealESRGAN_x2plus')
  })
})

// ------------------------------------------- the model a mode actually shows

/**
 * The Phase 10 P1 regression, at render level.
 *
 * Before this was fixed, Creative with an untouched dropdown showed
 * `Real-ESRGAN x4 Plus`, read `supportsDenoise` off it, disabled the
 * noise-reduction slider and rendered "Real-ESRGAN x4 Plus has no denoise
 * weights to blend." The job then ran `realesr-general-x4v3` at denoise 0.25 -
 * so the panel named the wrong model, made a false claim about it, and greyed
 * out the one control Creative exists to expose.
 *
 * Rendered rather than asserted against a helper, because the bug was entirely
 * in what the panel derived and a helper test would have passed throughout.
 */

const X4PLUS: ModelInfo = {
  id: 'RealESRGAN_x4plus',
  name: 'Real-ESRGAN x4 Plus',
  description: 'General-purpose 4x upscaler. Best default for photographs.',
  arch: 'RRDBNet',
  scale: 4,
  supportsDenoise: false,
  supportedScales: [4, 8, 16],
  downloaded: true,
  sizeMb: 63.9,
}

const GENERAL_V3: ModelInfo = {
  id: 'realesr-general-x4v3',
  name: 'realesr-general-x4v3',
  description: 'Compact 4x model.',
  arch: 'SRVGGNet',
  scale: 4,
  supportsDenoise: true,
  supportedScales: [4, 8, 16],
  downloaded: true,
  sizeMb: 4.7,
}

/** x4plus first, so the adopted default is the *wrong* model for Creative. */
function renderRail() {
  stubApi({ '/api/health': HEALTH, '/api/system': GPU_SYSTEM, '/api/models': [X4PLUS, GENERAL_V3] })
  useWorkspaceStore.setState({
    source: {
      file: new File([new Uint8Array(10)], 'p.png', { type: 'image/png' }),
      objectUrl: 'blob:t',
      metadata: { name: 'p.png', width: 1280, height: 720, sizeBytes: 10, format: 'PNG' },
    },
    problem: null,
    isLoading: false,
  })
  renderWithProviders(<EnhancePanels />)
}

/** The noise-reduction slider, which is disabled for a model with no pair. */
function denoiseSlider() {
  return screen.getByRole('slider', { name: /noise reduction/i })
}

async function chooseModel(name: string) {
  await userEvent.click(screen.getByRole('combobox', { name: /^model/i }))
  await userEvent.click(within(await screen.findByRole('listbox')).getByText(name))
}

describe('the model a mode shows before one is chosen', () => {
  it('shows the mode default table the backend agrees with', () => {
    // Guards the mirror itself. A contract test on the backend checks these
    // against `mode_planner`; this checks nothing silently emptied the table.
    expect(MODE_DEFAULT_MODEL.standard).toBe('RealESRGAN_x4plus')
    expect(MODE_DEFAULT_MODEL.creative).toBe('realesr-general-x4v3')
  })

  it('shows Creative running general-v3, not the adopted default', async () => {
    useEnhancementStore.setState({ mode: 'creative' })
    renderRail()

    const trigger = await screen.findByRole('combobox', { name: /^model/i })
    await waitFor(() => {
      expect(trigger).toHaveTextContent(GENERAL_V3.name)
    })
    expect(trigger).not.toHaveTextContent(X4PLUS.name)
  })

  it('enables the denoise slider in Creative, because that model has a pair', async () => {
    useEnhancementStore.setState({ mode: 'creative' })
    renderRail()

    await waitFor(() => {
      expect(denoiseSlider()).not.toHaveAttribute('data-disabled')
    })
  })

  it('no longer claims the Creative model has no denoise weights', async () => {
    useEnhancementStore.setState({ mode: 'creative' })
    renderRail()

    await screen.findByRole('combobox', { name: /^model/i })
    expect(screen.queryByText(/Real-ESRGAN x4 Plus has no denoise weights/i)).not.toBeInTheDocument()
  })

  it('describes the Creative model, not the adopted one', async () => {
    useEnhancementStore.setState({ mode: 'creative' })
    renderRail()
    await screen.findByRole('combobox', { name: /^model/i })

    // The per-model wording lives on the options, so the menu has to be open.
    await userEvent.click(screen.getByRole('combobox', { name: /^model/i }))
    const listbox = await screen.findByRole('listbox')

    const wording = i18n.getResource('en', 'models', `${GENERAL_V3.id}.description`) as string
    expect(within(listbox).getByText(wording)).toBeInTheDocument()
  })

  it('shows Standard running x4plus, with denoise off', async () => {
    useEnhancementStore.setState({ mode: 'standard' })
    renderRail()

    const trigger = await screen.findByRole('combobox', { name: /^model/i })
    await waitFor(() => {
      expect(trigger).toHaveTextContent(X4PLUS.name)
    })
    expect(denoiseSlider()).toHaveAttribute('data-disabled')
  })

  it.each([
    ['standard', 'creative', GENERAL_V3.name],
    ['creative', 'standard', X4PLUS.name],
  ] as const)('follows a %s to %s switch', async (from, to, expected) => {
    useEnhancementStore.setState({ mode: from })
    renderRail()
    await screen.findByRole('combobox', { name: /^model/i })

    useEnhancementStore.setState({ mode: to })

    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: /^model/i })).toHaveTextContent(expected)
    })
  })
})

describe('an explicitly chosen model', () => {
  it('overrides the mode, and its capabilities come with it', async () => {
    useEnhancementStore.setState({ mode: 'creative' })
    renderRail()
    await screen.findByRole('combobox', { name: /^model/i })

    await chooseModel(X4PLUS.name)

    expect(useEnhancementStore.getState().modelChosen).toBe(true)
    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: /^model/i })).toHaveTextContent(X4PLUS.name)
    })
    // x4plus has no pair, so the control is correctly disabled here - the same
    // state that was wrong when the mode had chosen the model.
    expect(denoiseSlider()).toHaveAttribute('data-disabled')
  })

  it('survives a mode switch rather than being overwritten by it', async () => {
    useEnhancementStore.setState({ mode: 'creative' })
    renderRail()
    await screen.findByRole('combobox', { name: /^model/i })
    await chooseModel(X4PLUS.name)

    useEnhancementStore.setState({ mode: 'standard' })

    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: /^model/i })).toHaveTextContent(X4PLUS.name)
    })
    expect(denoiseSlider()).toHaveAttribute('data-disabled')
  })

  it('keeps general-v3 and its enabled denoise when chosen from Standard', async () => {
    useEnhancementStore.setState({ mode: 'standard' })
    renderRail()
    await screen.findByRole('combobox', { name: /^model/i })

    await chooseModel(GENERAL_V3.name)

    await waitFor(() => {
      expect(denoiseSlider()).not.toHaveAttribute('data-disabled')
    })
  })
})
