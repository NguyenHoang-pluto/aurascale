import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ModeControls } from './ModeControls'
import { createJob } from '@/services/jobsApi'
import { renderWithProviders } from '@/test/renderWithProviders'
import {
  DEFAULT_MODE,
  DEFAULT_SCALE,
  DEFAULT_SIZING,
  useEnhancementStore,
} from '@/stores/useEnhancementStore'
import { TARGET_LONG_EDGE } from '@/types/job'
import type { CreateJobRequest } from '@/types/job'

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
