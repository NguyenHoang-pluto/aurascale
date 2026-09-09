import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { EnhancePanels } from './EnhancePanels'
import { buildSettings } from './useEnhanceJob'
import { describeBlocker, describeProgress, projectedSize, scaleOptions } from './jobPresentation'
import { mergeProgress } from './useJobProgress'
import { changeLanguage, i18n } from '@/i18n'
import { renderWithProviders } from '@/test/renderWithProviders'
import { jsonResponse } from '@/test/renderWithProviders'
import { GPU_SYSTEM, HEALTH, stubApi } from '@/test/systemFixtures'
import {
  DEFAULT_DENOISE,
  DEFAULT_QUALITY,
  DEFAULT_SCALE,
  useEnhancementStore,
} from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import type { JobRecord } from '@/types/job'
import type { ModelInfo } from '@/types/system'

/**
 * The enhancement rail, from picking a model to downloading a result.
 *
 * The backend is stubbed at `fetch`, so the real API client, the real stores
 * and the real components run. jsdom has no `EventSource`, which means these
 * exercise the polling fallback — the path a user behind an SSE-hostile proxy
 * gets, and the one that has to work without the stream.
 */

const MODELS: ModelInfo[] = [
  {
    id: 'RealESRGAN_x4plus',
    name: 'Real-ESRGAN x4 Plus',
    description: 'General-purpose 4x upscaler.',
    arch: 'RRDBNet',
    scale: 4,
    supportsDenoise: false,
    supportedScales: [4, 8],
    downloaded: true,
    sizeMb: 63.9,
  },
  {
    id: 'RealESRGAN_x2plus',
    name: 'Real-ESRGAN x2 Plus',
    description: 'Native 2x upscaler.',
    arch: 'RRDBNet',
    scale: 2,
    supportsDenoise: false,
    supportedScales: [2],
    downloaded: true,
    sizeMb: 64,
  },
  {
    id: 'realesr-general-x4v3',
    name: 'Real-ESRGAN General v3',
    description: 'Compact 4x model.',
    arch: 'SRVGGNetCompact',
    scale: 4,
    supportsDenoise: true,
    supportedScales: [4, 8],
    downloaded: true,
    sizeMb: 4.7,
  },
  {
    id: 'RealESRGAN_x4plus_anime_6B',
    name: 'Real-ESRGAN x4 Plus Anime',
    description: 'For illustration.',
    arch: 'RRDBNet',
    scale: 4,
    supportsDenoise: false,
    supportedScales: [4, 8],
    downloaded: false,
    sizeMb: null,
  },
]

const JOB_ID = 'a'.repeat(32)

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    jobId: JOB_ID,
    status: 'processing',
    stage: 'running_inference',
    progress: 52,
    model: 'RealESRGAN_x4plus',
    scale: 4,
    mode: null,
    outputType: 'scale',
    target: null,
    device: 'cuda',
    input: { width: 1280, height: 720, sizeBytes: 1_887_437, format: 'PNG' },
    output: null,
    processingMs: null,
    error: null,
    createdAt: '2026-09-08T10:00:00Z',
    startedAt: '2026-09-08T10:00:01Z',
    finishedAt: null,
    ...overrides,
  }
}

/** A backend that answers the registry and hands back a scripted job. */
function stubBackend(options: { jobs?: JobRecord[]; onCreate?: (body: FormData) => void } = {}) {
  const records = options.jobs ?? [job({ status: 'queued', stage: null, progress: 0 })]
  let index = 0
  const calls: { url: string; method: string; body?: unknown }[] = []

  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET'
    calls.push({ url, method, ...(init?.body !== undefined ? { body: init.body } : {}) })

    if (url.includes('/api/models')) return Promise.resolve(jsonResponse(MODELS))
    if (url.includes('/api/system')) return Promise.resolve(jsonResponse(GPU_SYSTEM))
    if (url.includes('/api/health')) return Promise.resolve(jsonResponse(HEALTH))

    if (url.includes('/api/jobs') && method === 'POST') {
      if (options.onCreate !== undefined) options.onCreate(init?.body as FormData)
      return Promise.resolve(
        jsonResponse(
          { jobId: JOB_ID, status: 'queued', queuePosition: 0, createdAt: '2026-09-08T10:00:00Z' },
          202,
        ),
      )
    }

    if (url.includes('/api/jobs') && method === 'DELETE') {
      return Promise.resolve(new Response(null, { status: 204 }))
    }

    if (url.includes(`/api/jobs/${JOB_ID}`)) {
      const record = records[Math.min(index, records.length - 1)]
      index += 1
      return Promise.resolve(jsonResponse(record))
    }

    return Promise.reject(new TypeError('Failed to fetch'))
  })

  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock, calls }
}

/** The value beside a labelled row in the job result list. */
function valueFor(label: string): string | undefined {
  const row = screen.getByText(label).closest('div')
  return row?.querySelector('dd')?.textContent ?? undefined
}

function loadImage(name = 'holiday.png') {
  useWorkspaceStore.setState({
    source: {
      file: new File([new Uint8Array(1000)], name, { type: 'image/png' }),
      objectUrl: 'blob:test',
      metadata: { name, width: 1280, height: 720, sizeBytes: 1_887_437, format: 'PNG' },
    },
    problem: null,
    isLoading: false,
  })
}

beforeEach(() => {
  useWorkspaceStore.setState({ source: null, problem: null, isLoading: false })
  useEnhancementStore.setState({
    modelId: null,
    scale: DEFAULT_SCALE,
    denoiseStrength: DEFAULT_DENOISE,
    sharpenStrength: 0,
    format: 'png',
    quality: DEFAULT_QUALITY,
    preserveMetadata: true,
    activeJobId: null,
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

// ------------------------------------------------------------------ controls

// ------------------------------------------------------------ initialisation

/**
 * One entry exactly as the running backend serialises it.
 *
 * Copied from a live `GET /api/models` rather than hand-written, so a field
 * the server renames or drops fails here instead of at the user.
 */
const LIVE_MODEL_PAYLOAD = {
  id: 'RealESRGAN_x4plus',
  name: 'Real-ESRGAN x4 Plus',
  description: 'General-purpose 4x upscaler. Best default for photographs.',
  arch: 'RRDBNet',
  scale: 4,
  supportsDenoise: false,
  supportedScales: [4, 8],
  downloaded: true,
  sizeMb: 63.9,
}

/** Answer only the model list; the rest of the shell is irrelevant here. */
function stubModels(payload: unknown) {
  return stubApi({ '/api/health': HEALTH, '/api/system': GPU_SYSTEM, '/api/models': payload })
}

describe('the root route initialising', () => {
  it('adopts a default from a model list exactly as the backend sends it', async () => {
    stubModels([LIVE_MODEL_PAYLOAD])
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(useEnhancementStore.getState().modelId).toBe('RealESRGAN_x4plus')
    })
    // 4x is on offer, so the default factor is kept rather than moved.
    expect(useEnhancementStore.getState().scale).toBe(4)
    expect(await screen.findByRole('combobox', { name: 'Model' })).toBeInTheDocument()
  })

  it('refuses a model list from an older backend instead of rendering nothing', async () => {
    // A dev server left running from a release before `supportedScales`
    // existed answers on the same port and returns exactly this.
    const { supportedScales: _omitted, ...stale } = LIVE_MODEL_PAYLOAD
    stubModels([stale])

    renderWithProviders(<EnhancePanels />)

    // The panel says so; it does not throw on the way to the store.
    expect(await screen.findByText('Cannot load the model list')).toBeInTheDocument()
    expect(useEnhancementStore.getState().modelId).toBeNull()
  })

  it('refuses a list whose scales are not numbers', async () => {
    stubModels([{ ...LIVE_MODEL_PAYLOAD, supportedScales: ['4', '8'] }])
    renderWithProviders(<EnhancePanels />)

    expect(await screen.findByText('Cannot load the model list')).toBeInTheDocument()
    expect(useEnhancementStore.getState().modelId).toBeNull()
  })

  it('refuses a body that is not a list at all', async () => {
    stubModels({ models: [LIVE_MODEL_PAYLOAD] })
    renderWithProviders(<EnhancePanels />)

    expect(await screen.findByText('Cannot load the model list')).toBeInTheDocument()
    expect(useEnhancementStore.getState().modelId).toBeNull()
  })

  it('accepts an empty registry, which is a real state and not an error', async () => {
    // A backend with no weights installed reports no models. Nothing to adopt,
    // and nothing to complain about.
    stubModels([])
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.queryByRole('combobox', { name: 'Model' })).toBeInTheDocument()
    })
    expect(screen.queryByText('Cannot load the model list')).not.toBeInTheDocument()
    expect(useEnhancementStore.getState().modelId).toBeNull()
  })

  it('accepts a model that can produce no factor, keeping the current one', async () => {
    // An empty `supportedScales` is describable: the model is unusable, every
    // factor renders disabled, and the stored scale is left alone.
    stubModels([{ ...LIVE_MODEL_PAYLOAD, supportedScales: [] }])
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(useEnhancementStore.getState().modelId).toBe('RealESRGAN_x4plus')
    })
    expect(useEnhancementStore.getState().scale).toBe(DEFAULT_SCALE)
    expect(screen.queryByText('Cannot load the model list')).not.toBeInTheDocument()
  })

  it('shows the loading state while the registry is still in flight', () => {
    // Never resolves: the panel must render its skeleton, not crash on
    // `undefined` data.
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(() => undefined)))
    const { container } = renderWithProviders(<EnhancePanels />)

    expect(container.querySelector('[aria-busy="true"]')).toBeInTheDocument()
    expect(useEnhancementStore.getState().modelId).toBeNull()
  })
})

describe('the enhancement controls', () => {
  it('offers the models the backend reports', async () => {
    stubBackend()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeInTheDocument()
    })
    // A model is selected as soon as the registry is known, so the panel never
    // shows an empty control.
    expect(screen.getByLabelText('Model')).toHaveTextContent('Real-ESRGAN x4 Plus')
  })

  it('shows a skeleton while the registry is loading', () => {
    stubBackend()
    const { container } = renderWithProviders(<EnhancePanels />)

    expect(container.querySelector('[aria-busy="true"]')).toBeInTheDocument()
  })

  it('reports an unreachable backend instead of an empty model list', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))))
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByText('Cannot load the model list')).toBeInTheDocument()
    })
  })

  it('enables the denoise control only for a model that has the weights', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeInTheDocument()
    })

    // x4plus has no denoise pair.
    const slider = screen.getByRole('slider', { name: 'Noise reduction' })
    expect(slider).toHaveAttribute('data-disabled')

    await user.click(screen.getByLabelText('Model'))
    await user.click(screen.getByRole('option', { name: /General v3/ }))

    await waitFor(() => {
      expect(screen.getByRole('slider', { name: 'Noise reduction' })).not.toHaveAttribute(
        'data-disabled',
      )
    })
  })

  it('never offers a scale the selected model cannot produce', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeInTheDocument()
    })

    await user.click(screen.getByLabelText('Model'))
    await user.click(screen.getByRole('option', { name: /x2 Plus/ }))

    // The 2x model can only do 2x, so 4x and 8x are visibly unavailable.
    await waitFor(() => {
      expect(screen.getByRole('radio', { name: /^2x/ })).toBeEnabled()
    })
    expect(screen.getByRole('radio', { name: /^4x/ })).toBeDisabled()
    expect(screen.getByRole('radio', { name: /8x/ })).toBeDisabled()
  })

  it('moves the scale to something the new model supports', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeInTheDocument()
    })
    await user.click(screen.getByRole('radio', { name: /8x/ }))
    expect(useEnhancementStore.getState().scale).toBe(8)

    await user.click(screen.getByLabelText('Model'))
    await user.click(screen.getByRole('option', { name: /x2 Plus/ }))

    // 8x is impossible for this model, so the UI does not sit on it.
    await waitFor(() => {
      expect(useEnhancementStore.getState().scale).toBe(2)
    })
  })

  it('marks a model whose weights are missing as unusable', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeInTheDocument()
    })
    await user.click(screen.getByLabelText('Model'))

    expect(screen.getByRole('option', { name: /Anime/ })).toHaveAttribute('data-disabled')
  })
})

// -------------------------------------------------------------------- output

// -------------------------------------------------------------- localisation

describe('model descriptions in the selector', () => {
  /**
   * Open the model dropdown and return the option list.
   *
   * The descriptions live inside the options, which Radix only mounts once the
   * trigger is opened.
   */
  async function openModelList(user: ReturnType<typeof userEvent.setup>) {
    // The field's own label is translated too, so it is looked up rather than
    // spelled — "Model" in English, "Mô hình" in Vietnamese.
    const label = i18n.t('enhance:model.label')

    await waitFor(() => {
      expect(screen.getByLabelText(label)).toBeInTheDocument()
    })
    await user.click(screen.getByLabelText(label))
    return within(await screen.findByRole('listbox'))
  }

  it('describes each model in English', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    const list = await openModelList(user)

    expect(
      list.getByText('General-purpose 4x upscaler. Best default for photographs.'),
    ).toBeInTheDocument()
    expect(list.getByText(/Native 2x upscaler/)).toBeInTheDocument()
  })

  it('describes each model in Vietnamese', async () => {
    changeLanguage('vi')
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    const list = await openModelList(user)

    expect(list.getByText(/Mô hình phóng 4x đa dụng/)).toBeInTheDocument()
    // The English the backend sent is gone from the screen entirely.
    expect(
      list.queryByText('General-purpose 4x upscaler. Best default for photographs.'),
    ).not.toBeInTheDocument()
  })

  it('updates the description when the language changes, with no reload', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    let list = await openModelList(user)
    expect(list.getByText(/General-purpose 4x upscaler/)).toBeInTheDocument()
    await user.keyboard('{Escape}')

    // The same mounted tree, only the language moved.
    changeLanguage('vi')

    list = await openModelList(user)
    await waitFor(() => {
      expect(list.getByText(/Mô hình phóng 4x đa dụng/)).toBeInTheDocument()
    })
    expect(list.queryByText(/General-purpose 4x upscaler/)).not.toBeInTheDocument()
  })

  it('leaves the model names untranslated in either language', async () => {
    changeLanguage('vi')
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    const list = await openModelList(user)

    // Names match the manifest and the weights files, so they do not move.
    for (const name of [
      'Real-ESRGAN x4 Plus',
      'Real-ESRGAN x2 Plus',
      'Real-ESRGAN General v3',
      'Real-ESRGAN x4 Plus Anime',
    ]) {
      expect(list.getByText(name), name).toBeInTheDocument()
    }
  })

  it('prefers its own wording over whatever sentence the server sent', async () => {
    // The server's description is deliberately different from the locale file.
    // If the UI still passed the payload straight through, this would show.
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string) => {
        if (url.includes('/api/models')) {
          return Promise.resolve(
            jsonResponse([
              { ...MODELS[0], description: 'SERVER TEXT THAT MUST NOT BE RENDERED' },
            ]),
          )
        }
        if (url.includes('/api/system')) return Promise.resolve(jsonResponse(GPU_SYSTEM))
        if (url.includes('/api/health')) return Promise.resolve(jsonResponse(HEALTH))
        return Promise.reject(new TypeError('Failed to fetch'))
      }),
    )

    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    const list = await openModelList(user)

    expect(list.getByText(/General-purpose 4x upscaler/)).toBeInTheDocument()
    expect(list.queryByText(/SERVER TEXT/)).not.toBeInTheDocument()
  })

  it('translates the description of a model whose weights are missing', async () => {
    changeLanguage('vi')
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    const list = await openModelList(user)

    // The anime model is the undownloaded one in this fixture; its description
    // is wrapped in the "not downloaded" phrasing, and both halves translate.
    const anime = list.getByText(/Biến thể 6 khối/)
    expect(anime).toBeInTheDocument()
    expect(anime.textContent).toContain('chưa tải về')
  })
})

describe('the output controls', () => {
  it('disables quality for a lossless format', async () => {
    stubBackend()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('slider', { name: 'Output quality' })).toBeInTheDocument()
    })
    expect(screen.getByRole('slider', { name: 'Output quality' })).toHaveAttribute('data-disabled')
  })

  it('enables quality once a lossy format is chosen', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('radio', { name: /JPEG/ })).toBeInTheDocument()
    })
    await user.click(screen.getByRole('radio', { name: /JPEG/ }))

    await waitFor(() => {
      expect(screen.getByRole('slider', { name: 'Output quality' })).not.toHaveAttribute(
        'data-disabled',
      )
    })
  })

  it('states the size the result will be', async () => {
    stubBackend()
    loadImage()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByText('5,120 × 2,880')).toBeInTheDocument()
    })
  })
})

// ------------------------------------------------------------------- the job

describe('submitting a job', () => {
  it('will not submit without an image, and says why', async () => {
    stubBackend()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeDisabled()
    })
    expect(screen.getByText('Load an image to enhance.')).toBeInTheDocument()
  })

  it('sends the chosen settings as a real multipart submission', async () => {
    let submitted: FormData | undefined
    stubBackend({ onCreate: (body) => { submitted = body } })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    await waitFor(() => {
      expect(submitted).toBeDefined()
    })
    const form = submitted as unknown as FormData
    expect(form.get('model')).toBe('RealESRGAN_x4plus')
    expect(form.get('scale')).toBe('4')
    expect(form.get('format')).toBe('png')
    expect(form.get('preserveMetadata')).toBe('true')
    expect(form.get('image')).toBeInstanceOf(File)
    // PNG is lossless, so no quality is sent at all.
    expect(form.get('quality')).toBeNull()
  })

  it('reports a rejected submission without pretending it started', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((url: string, init?: RequestInit) => {
        if (url.includes('/api/models')) return Promise.resolve(jsonResponse(MODELS))
        if ((init?.method ?? 'GET') === 'POST') {
          return Promise.resolve(
            jsonResponse(
              {
                type: 'https://pixelforge.ai/errors/image_too_large',
                title: 'Image too large',
                status: 413,
                code: 'image_too_large',
                // The English detail is the server's; the UI builds its own
                // sentence from the code and this context.
                detail: 'That image is 81 MP, which is larger than the 16 MP limit.',
                context: { actualPixels: 81_000_000, limitPixels: 16_000_000 },
              },
              413,
            ),
          )
        }
        return Promise.reject(new TypeError('Failed to fetch'))
      }),
    )
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    await waitFor(() => {
      expect(screen.getByText(/larger than the 16.0 MP limit/)).toBeInTheDocument()
    })
    expect(useEnhancementStore.getState().activeJobId).toBeNull()
  })

  it('follows a job to completion and offers the result', async () => {
    stubBackend({
      jobs: [
        job({ status: 'queued', stage: null, progress: 0 }),
        job({ status: 'processing', progress: 52 }),
        job({
          status: 'completed',
          stage: null,
          progress: 100,
          output: { width: 5120, height: 2880, sizeBytes: 9_122_611, format: 'PNG' },
          processingMs: 8420,
          finishedAt: '2026-09-08T10:00:09Z',
        }),
      ],
    })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    // The measured result, not an assumption about what was requested.
    await waitFor(
      () => {
        expect(screen.getByText('Completed')).toBeInTheDocument()
      },
      { timeout: 5000 },
    )

    // Read row by row: the output controls project the same size from the
    // settings, and what matters here is the figure the server reported.
    expect(valueFor('Result')).toBe('5,120 × 2,880')
    expect(valueFor('File size')).toBe('8.7 MB')
    expect(valueFor('Took')).toBe('8.42s')

    const download = screen.getByRole('link', { name: /Download result/ })
    expect(download).toHaveAttribute('href', expect.stringContaining(`/api/jobs/${JOB_ID}/result`))
    expect(download).toHaveAttribute('download')
  })

  it('surfaces a failed job with its reason', async () => {
    stubBackend({
      jobs: [
        job({
          status: 'failed',
          stage: null,
          error: {
            code: 'out_of_memory',
            detail: 'There is not enough memory to enhance an image this large.',
            technical: 'CUDA out of memory',
          },
        }),
      ],
    })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    await waitFor(() => {
      expect(screen.getByText(/not enough memory/)).toBeInTheDocument()
    })
    expect(screen.queryByRole('link', { name: /Download/ })).not.toBeInTheDocument()
  })

  it('offers no result for a cancelled job', async () => {
    stubBackend({ jobs: [job({ status: 'cancelled', stage: null, progress: 0 })] })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    await waitFor(() => {
      expect(screen.getByText('Cancelled')).toBeInTheDocument()
    })
    expect(screen.getByText(/no result was produced/)).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Download/ })).not.toBeInTheDocument()
  })

  it('sends a cancel request for a running job', async () => {
    const { calls } = stubBackend({ jobs: [job({ status: 'processing' })] })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Cancel/ })).toBeInTheDocument()
    })
    await user.click(screen.getByRole('button', { name: /Cancel/ }))

    await waitFor(() => {
      expect(calls.some((call) => call.method === 'DELETE')).toBe(true)
    })
  })

  it('locks the settings while a job is running', async () => {
    stubBackend({ jobs: [job({ status: 'processing' })] })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    // Changing the model mid-job would describe a job that is not the one
    // running, so the controls are disabled rather than merely ignored.
    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeDisabled()
    })
  })

  it('shows the stage and the real tile count while it runs', async () => {
    stubBackend({ jobs: [job({ status: 'processing', progress: 52 })] })
    loadImage()
    const user = userEvent.setup()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enhance' })).toBeEnabled()
    })
    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    await waitFor(() => {
      expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '52')
    })
    expect(screen.getByText('Enhancing')).toBeInTheDocument()
  })
})

// ----------------------------------------------------------- pure behaviour

describe('scaleOptions', () => {
  it('disables everything the model cannot produce', () => {
    const options = scaleOptions(i18n.t, [2])

    expect(options.map((option) => [option.value, option.disabled])).toEqual([
      ['2', false],
      ['4', true],
      ['8', true],
      ['16', true],
    ])
  })

  it('labels 8x as the two-pass route it is', () => {
    expect(scaleOptions(i18n.t, [4, 8]).find((option) => option.value === '8')?.hint).toBe('two-pass')
  })
})

describe('projectedSize', () => {
  it('multiplies the input by the factor', () => {
    expect(projectedSize({ name: 'x', width: 100, height: 50, sizeBytes: 1, format: 'PNG' }, 4))
      .toEqual({ width: 400, height: 200 })
  })

  it('has nothing to project without an image', () => {
    expect(projectedSize(undefined, 4)).toBeUndefined()
  })
})

describe('buildSettings', () => {
  it('omits denoise for a model that cannot use it', () => {
    const settings = buildSettings({
      sharpenStrength: 0.5,
      denoiseStrength: 0.5,
      supportsDenoise: false,
    })

    expect(settings).toEqual({ sharpenStrength: 0.5 })
  })

  it('sends denoise for a model that can', () => {
    const settings = buildSettings({
      sharpenStrength: 0,
      denoiseStrength: 0.25,
      supportsDenoise: true,
    })

    expect(settings).toEqual({ denoiseStrength: 0.25 })
  })

  it('omits sharpening when it is off, rather than sending zero', () => {
    expect(
      buildSettings({ sharpenStrength: 0, denoiseStrength: 1, supportsDenoise: false }),
    ).toEqual({})
  })
})

describe('describeProgress', () => {
  const live = { progress: 52, stage: 'running_inference' as const, tilesDone: 1, tilesTotal: 2 }

  it('names the tile during inference, because that is what the number means', () => {
    expect(describeProgress(i18n.t, live, 'processing')).toBe('Enhancing · tile 1 of 2')
  })

  it('names only the stage when there are no tiles to count', () => {
    expect(describeProgress(i18n.t, { ...live, tilesDone: null, tilesTotal: null }, 'processing')).toBe(
      'Enhancing',
    )
  })

  it('says a queued job is waiting rather than working', () => {
    expect(describeProgress(i18n.t, live, 'queued')).toBe('Waiting for a free worker')
  })
})

describe('mergeProgress', () => {
  const streamed = { progress: 90, stage: 'running_inference' as const, tilesDone: 2, tilesTotal: 2 }

  it('prefers the stream while a job runs, since it reports first', () => {
    expect(mergeProgress(job({ progress: 52 }), streamed).progress).toBe(90)
  })

  it('keeps the record when the stream is behind it', () => {
    expect(mergeProgress(job({ progress: 95 }), { ...streamed, progress: 15 }).progress).toBe(95)
  })

  it('shows a completed job at 100, not at the last tile', () => {
    const merged = mergeProgress(job({ status: 'completed', progress: 100 }), streamed)

    expect(merged.progress).toBe(100)
    expect(merged.stage).toBeNull()
  })

  it('does not invent progress before anything has been measured', () => {
    expect(mergeProgress(undefined, null)).toEqual({
      progress: 0,
      stage: null,
      tilesDone: null,
      tilesTotal: null,
    })
  })
})

describe('describeBlocker', () => {
  it('asks for an image first', () => {
    expect(describeBlocker(i18n.t, false, undefined)).toBe('Load an image to enhance.')
  })

  it('waits for the registry before blaming anything else', () => {
    expect(describeBlocker(i18n.t, true, undefined)).toBe('Waiting for the model list.')
  })

  it('names a model whose weights are missing', () => {
    expect(describeBlocker(i18n.t, true, { downloaded: false, name: 'Anime' })).toBe(
      'Anime is not downloaded yet.',
    )
  })

  it('has nothing to say when everything is ready', () => {
    expect(describeBlocker(i18n.t, true, { downloaded: true, name: 'x4plus' })).toBeUndefined()
  })
})

describe('accessibility', () => {
  it('labels every control in the rail', async () => {
    stubBackend()
    loadImage()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByLabelText('Model')).toBeInTheDocument()
    })

    for (const name of ['Noise reduction', 'Sharpening strength', 'Output quality']) {
      expect(screen.getByRole('slider', { name })).toBeInTheDocument()
    }
    for (const group of ['Upscale factor', 'Output format']) {
      expect(screen.getByRole('radiogroup', { name: group })).toBeInTheDocument()
    }
    expect(screen.getByRole('switch', { name: /Preserve metadata/ })).toBeInTheDocument()
  })

  it('marks the unbuilt control as unbuilt rather than hiding it', async () => {
    stubBackend()
    renderWithProviders(<EnhancePanels />)

    await waitFor(() => {
      expect(screen.getByText('Artifact reduction')).toBeInTheDocument()
    })
    const row = screen.getByText('Artifact reduction').closest('div')
    expect(within(row as HTMLElement).getByText('Coming soon')).toBeInTheDocument()
    expect(screen.getByRole('switch', { name: /Artifact reduction/ })).toBeDisabled()
  })
})
