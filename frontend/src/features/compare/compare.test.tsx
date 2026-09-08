import { act, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ComparisonViewer } from './ComparisonViewer'
import { ResultCanvas } from './ResultCanvas'
import { SPLIT_POSITION } from './comparisonMath'
import { useCropLayer } from './useCropLayer'
import { EnhancePage } from '@/pages/EnhancePage'
import { renderWithProviders, jsonResponse } from '@/test/renderWithProviders'
import { stubBoundingRect } from '@/test/browserStubs'
import { GPU_SYSTEM, HEALTH } from '@/test/systemFixtures'
import { useComparisonStore } from '@/stores/useComparisonStore'
import { useEnhancementStore } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import type { JobRecord } from '@/types/job'
import type { ModelInfo } from '@/types/system'
import type { Transform } from '@/features/viewer/zoomMath'

/**
 * The comparison surface.
 *
 * jsdom lays nothing out, so container sizes are stubbed where a test needs
 * the crop layer to have a viewport to reason about. Everything else - the
 * stores, the components, the URL building - runs for real.
 */

const JOB_ID = 'b'.repeat(32)
const OUTPUT = { width: 1280, height: 960 }
/** A viewport the crop probe can reason about, since jsdom reports none. */
const PROBE_CONTAINER = { width: 800, height: 600 }

const BEFORE = {
  objectUrl: 'blob:before',
  width: 320,
  height: 240,
  name: 'holiday.png',
}

const MODELS: ModelInfo[] = [
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
]

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    jobId: JOB_ID,
    status: 'completed',
    stage: null,
    progress: 100,
    model: 'realesr-general-x4v3',
    scale: 4,
    device: 'cuda',
    input: { width: 320, height: 240, sizeBytes: 161_035, format: 'PNG' },
    output: { width: 1280, height: 960, sizeBytes: 1_018_871, format: 'PNG' },
    processingMs: 8420,
    error: null,
    createdAt: '2026-09-08T10:00:00Z',
    startedAt: '2026-09-08T10:00:01Z',
    finishedAt: '2026-09-08T10:00:09Z',
    ...overrides,
  }
}

/** A backend answering the registry, one job record, and crop requests. */
function stubBackend(record: JobRecord = job()) {
  const cropRequests: string[] = []

  const fetchMock = vi.fn((url: string) => {
    if (url.includes('/api/models')) return Promise.resolve(jsonResponse(MODELS))
    if (url.includes('/api/system')) return Promise.resolve(jsonResponse(GPU_SYSTEM))
    if (url.includes('/api/health')) return Promise.resolve(jsonResponse(HEALTH))

    if (url.includes('/preview?')) {
      cropRequests.push(url)
      return Promise.resolve(new Response(new Blob([new Uint8Array([1, 2, 3])])))
    }
    if (url.includes(`/api/jobs/${JOB_ID}`)) return Promise.resolve(jsonResponse(record))

    return Promise.reject(new TypeError('Failed to fetch'))
  })

  vi.stubGlobal('fetch', fetchMock)
  return { fetchMock, cropRequests }
}

function renderViewer(props: Partial<Parameters<typeof ComparisonViewer>[0]> = {}) {
  return renderWithProviders(
    <ComparisonViewer jobId={JOB_ID} before={BEFORE} output={OUTPUT} {...props} />,
  )
}

beforeEach(() => {
  useComparisonStore.setState({ mode: 'slider', divider: SPLIT_POSITION, before: BEFORE })
  useEnhancementStore.setState({ activeJobId: null })
  useWorkspaceStore.setState({ source: null, problem: null, isLoading: false })
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

// ------------------------------------------------------------------- layers

describe('the comparison surface', () => {
  it('shows the original and the enhanced result together', () => {
    stubBackend()
    renderViewer()

    expect(screen.getByAltText(`Original: ${BEFORE.name}`)).toBeInTheDocument()
    expect(screen.getByAltText('Enhanced result')).toBeInTheDocument()
  })

  it('loads the capped preview, never the full result', () => {
    stubBackend()
    renderViewer()

    const after = screen.getByAltText('Enhanced result')
    expect(after).toHaveAttribute('src', expect.stringContaining(`/api/jobs/${JOB_ID}/preview`))
    expect(after.getAttribute('src')).not.toContain('/result')
  })

  it('draws both layers in the output coordinate space', () => {
    // The original is 320x240 and the result 1280x960; laying them out at
    // their own sizes would put different parts of the picture on either side
    // of the divider.
    stubBackend()
    renderViewer()

    for (const alt of [`Original: ${BEFORE.name}`, 'Enhanced result']) {
      const layer = screen.getByAltText(alt)
      expect(layer).toHaveStyle({ width: '1280px', height: '960px' })
    }
  })

  it('names the viewer and its keyboard shortcuts', () => {
    stubBackend()
    renderViewer()

    expect(screen.getByRole('group', { name: /Comparison viewer/ })).toBeInTheDocument()
  })
})

// -------------------------------------------------------------------- modes

describe('comparison modes', () => {
  it('offers all three', () => {
    stubBackend()
    renderViewer()

    const group = screen.getByRole('radiogroup', { name: 'Comparison mode' })
    for (const label of ['Slider', 'Side by side', 'Split']) {
      expect(within(group).getByRole('radio', { name: new RegExp(label) })).toBeInTheDocument()
    }
  })

  it('shows two panes side by side, each labelled', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('radio', { name: /Side by side/ }))

    expect(screen.getByRole('region', { name: 'Original' })).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Enhanced' })).toBeInTheDocument()
  })

  it('drives both panes from one transform', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderViewer()
    await user.click(screen.getByRole('radio', { name: /Side by side/ }))

    // Zoom, then check the two layers still agree pixel for pixel.
    await user.click(screen.getByRole('button', { name: /Zoom in/ }))

    await waitFor(() => {
      const before = screen.getByAltText(`Original: ${BEFORE.name}`)
      const after = screen.getByAltText('Enhanced result')
      expect(before.style.transform).toBe(after.style.transform)
      expect(before.style.transform).not.toBe('')
    })
  })

  it('keeps the transform when the mode changes', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('button', { name: /Zoom in/ }))
    const zoomed = screen.getByAltText('Enhanced result').style.transform

    await user.click(screen.getByRole('radio', { name: /Side by side/ }))
    await user.click(screen.getByRole('radio', { name: /^Slider/ }))

    expect(screen.getByAltText('Enhanced result').style.transform).toBe(zoomed)
  })

  it('fixes the split at the halfway point with no handle to drag', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('radio', { name: /Split/ }))

    expect(screen.queryByRole('slider', { name: 'Comparison divider' })).not.toBeInTheDocument()
  })

  it('ignores a stored divider position in split mode', async () => {
    stubBackend()
    useComparisonStore.setState({ divider: 12 })
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('radio', { name: /Split/ }))

    // The clip still reads 50%: split means split.
    const clipped = screen.getByAltText('Enhanced result').parentElement
    expect(clipped).toHaveStyle({ clipPath: 'inset(0 50% 0 0)' })
  })
})

// ------------------------------------------------------------------ divider

describe('the slider divider', () => {
  it('is a labelled slider with its position announced', () => {
    stubBackend()
    renderViewer()

    const divider = screen.getByRole('slider', { name: 'Comparison divider' })
    expect(divider).toHaveAttribute('aria-valuenow', '50')
    expect(divider).toHaveAttribute('aria-valuemin', '0')
    expect(divider).toHaveAttribute('aria-valuemax', '100')
    expect(divider).toHaveAttribute('aria-valuetext', '50% enhanced')
  })

  it('moves with the arrow keys', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderViewer()

    const divider = screen.getByRole('slider', { name: 'Comparison divider' })
    divider.focus()
    await user.keyboard('{ArrowRight}')

    expect(useComparisonStore.getState().divider).toBe(52)
  })

  it('jumps to the edges with Home and End', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderViewer()

    const divider = screen.getByRole('slider', { name: 'Comparison divider' })
    divider.focus()

    await user.keyboard('{Home}')
    expect(useComparisonStore.getState().divider).toBe(0)

    await user.keyboard('{End}')
    expect(useComparisonStore.getState().divider).toBe(100)
  })

  it('clamps rather than running past an edge', async () => {
    stubBackend()
    useComparisonStore.setState({ divider: 99 })
    const user = userEvent.setup()
    renderViewer()

    const divider = screen.getByRole('slider', { name: 'Comparison divider' })
    divider.focus()
    await user.keyboard('{ArrowRight}{ArrowRight}{ArrowRight}')

    expect(useComparisonStore.getState().divider).toBe(100)
  })

  it('clips the enhanced layer at the divider', () => {
    stubBackend()
    useComparisonStore.setState({ divider: 30 })
    renderViewer()

    const clipped = screen.getByAltText('Enhanced result').parentElement
    expect(clipped).toHaveStyle({ clipPath: 'inset(0 70% 0 0)' })
  })
})

// --------------------------------------------------------------- crop layer

describe('the full-resolution crop layer', () => {
  it('asks for nothing at or below 100% zoom', async () => {
    const { cropRequests } = stubBackend()
    vi.useFakeTimers()
    const { container } = renderViewer()
    stubBoundingRect(container.querySelector('[role="group"]') as Element, 800, 600)

    await act(async () => {
      vi.advanceTimersByTime(1000)
      await Promise.resolve()
    })

    expect(cropRequests).toEqual([])
  })

  it('requests one crop after the view settles above 100%', async () => {
    const { cropRequests } = stubBackend()
    const { rerender } = renderWithProviders(
      <CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />,
    )
    rerender(<CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />)

    await waitFor(() => {
      expect(cropRequests).toHaveLength(1)
    })
    expect(cropRequests[0]).toContain('x=0&y=0&w=400&h=300')
  })

  it('collapses a burst of movement into a single request', async () => {
    const { cropRequests } = stubBackend()
    const { rerender } = renderWithProviders(
      <CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />,
    )

    // Several frames of a pan, faster than the debounce.
    for (const tx of [-10, -20, -30, -40]) {
      rerender(<CropProbe transform={{ scale: 2, tx, ty: 0 }} />)
    }

    await waitFor(() => {
      expect(cropRequests.length).toBeGreaterThan(0)
    })
    expect(cropRequests).toHaveLength(1)
    // The region that was asked for is the one the pan ended on.
    expect(cropRequests[0]).toContain('x=20')
  })

  it('does not refetch a region it already holds', async () => {
    const { cropRequests } = stubBackend()
    const { rerender } = renderWithProviders(
      <CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />,
    )
    await waitFor(() => {
      expect(cropRequests).toHaveLength(1)
    })

    rerender(<CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />)
    await new Promise((resolve) => setTimeout(resolve, 400))

    expect(cropRequests).toHaveLength(1)
  })

  it('revokes the previous crop when a new one replaces it', async () => {
    const { cropRequests } = stubBackend()
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    const { rerender } = renderWithProviders(
      <CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />,
    )
    await waitFor(() => {
      expect(cropRequests).toHaveLength(1)
    })

    rerender(<CropProbe transform={{ scale: 2, tx: -600, ty: 0 }} />)
    await waitFor(() => {
      expect(cropRequests).toHaveLength(2)
    })

    expect(revoke).toHaveBeenCalled()
  })

  it('releases the crop when the viewer goes away', async () => {
    const { cropRequests } = stubBackend()
    const revoke = vi.spyOn(URL, 'revokeObjectURL')
    const { unmount } = renderWithProviders(
      <CropProbe transform={{ scale: 2, tx: 0, ty: 0 }} />,
    )
    await waitFor(() => {
      expect(cropRequests).toHaveLength(1)
    })

    unmount()

    expect(revoke).toHaveBeenCalled()
  })
})

// ------------------------------------------------------------------- states

describe('states', () => {
  it('says so quietly while the result is loading', () => {
    stubBackend()
    renderViewer()

    expect(screen.getByRole('status')).toHaveTextContent('Loading result')
    // The original stays on screen throughout.
    expect(screen.getByAltText(`Original: ${BEFORE.name}`)).toBeInTheDocument()
  })

  it('stops saying so once the result has loaded', async () => {
    stubBackend()
    renderViewer()

    const after = screen.getByAltText('Enhanced result')
    await act(async () => {
      after.dispatchEvent(new Event('load'))
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.queryByRole('status')).not.toBeInTheDocument()
    })
  })

  it('keeps the original usable and the download reachable when the preview fails', async () => {
    stubBackend()
    renderWithProviders(
      <ResultCanvas result={{ jobId: JOB_ID, before: BEFORE, output: OUTPUT }} />,
    )

    const after = screen.getByAltText('Enhanced result')
    await act(async () => {
      after.dispatchEvent(new Event('error'))
      await Promise.resolve()
    })

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('could not be loaded')
    })
    expect(screen.getByAltText(`Original: ${BEFORE.name}`)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: /Download result/ })).toHaveAttribute(
      'href',
      expect.stringContaining(`/api/jobs/${JOB_ID}/result`),
    )
  })

  it('does not blame a missing job for a preview it could not load', async () => {
    // Regression: the panel used to hardcode code="job_not_found". An <img>
    // onError says the browser could not load the preview and nothing more -
    // not the status, not a problem document - so naming a cause was a guess,
    // and it was the wrong one. An 8x result that exists and downloads fine
    // was reported as a deleted job.
    stubBackend()
    renderWithProviders(
      <ResultCanvas result={{ jobId: JOB_ID, before: BEFORE, output: OUTPUT }} />,
    )

    await act(async () => {
      screen.getByAltText('Enhanced result').dispatchEvent(new Event('error'))
      await Promise.resolve()
    })

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent('could not be loaded')

    // No invented code, and therefore no technical disclosure claiming one.
    expect(alert).not.toHaveTextContent('job_not_found')
    expect(alert).not.toHaveTextContent(/code:/i)
    expect(
      within(alert).queryByRole('button', { name: /Technical details/i }),
    ).not.toBeInTheDocument()
  })

  it('offers a retry after a failed preview', async () => {
    stubBackend()
    renderWithProviders(
      <ResultCanvas result={{ jobId: JOB_ID, before: BEFORE, output: OUTPUT }} />,
    )
    await act(async () => {
      screen.getByAltText('Enhanced result').dispatchEvent(new Event('error'))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument()
    })

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Try again' }))

    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })
})

// ------------------------------------------------------------------- gating

describe('when the comparison appears', () => {
  function loadImage() {
    useWorkspaceStore.setState({
      source: {
        file: new File([new Uint8Array(10)], 'holiday.png', { type: 'image/png' }),
        objectUrl: 'blob:source',
        metadata: {
          name: 'holiday.png',
          width: 320,
          height: 240,
          sizeBytes: 161_035,
          format: 'PNG',
        },
      },
      problem: null,
      isLoading: false,
    })
  }

  it('stays out of the way until a job has completed', async () => {
    stubBackend(job({ status: 'processing', stage: 'running_inference', progress: 52 }))
    loadImage()
    useEnhancementStore.setState({ activeJobId: JOB_ID })
    renderWithProviders(<EnhancePage />)

    await waitFor(() => {
      expect(screen.getByRole('group', { name: /Image viewer/ })).toBeInTheDocument()
    })
    expect(screen.queryByRole('radiogroup', { name: 'Comparison mode' })).not.toBeInTheDocument()
  })

  it('replaces the viewer once the result exists', async () => {
    stubBackend()
    loadImage()
    useEnhancementStore.setState({ activeJobId: JOB_ID })
    renderWithProviders(<EnhancePage />)

    await waitFor(() => {
      expect(screen.getByRole('radiogroup', { name: 'Comparison mode' })).toBeInTheDocument()
    })
    expect(screen.queryByRole('group', { name: /Image viewer/ })).not.toBeInTheDocument()
  })

  it('never compares a failed job, which has no result', async () => {
    stubBackend(
      job({
        status: 'failed',
        output: null,
        error: { code: 'out_of_memory', detail: 'Not enough memory.' },
      }),
    )
    loadImage()
    useEnhancementStore.setState({ activeJobId: JOB_ID })
    renderWithProviders(<EnhancePage />)

    await waitFor(() => {
      expect(screen.getByRole('group', { name: /Image viewer/ })).toBeInTheDocument()
    })
    expect(screen.queryByRole('radiogroup', { name: 'Comparison mode' })).not.toBeInTheDocument()
  })

  it('never compares a cancelled job', async () => {
    stubBackend(job({ status: 'cancelled', output: null, progress: 0 }))
    loadImage()
    useEnhancementStore.setState({ activeJobId: JOB_ID })
    renderWithProviders(<EnhancePage />)

    await waitFor(() => {
      expect(screen.getByRole('group', { name: /Image viewer/ })).toBeInTheDocument()
    })
    expect(screen.queryByRole('radiogroup', { name: 'Comparison mode' })).not.toBeInTheDocument()
  })

  it('shows the plain viewer when no job has been submitted', async () => {
    stubBackend()
    loadImage()
    renderWithProviders(<EnhancePage />)

    await waitFor(() => {
      expect(screen.getByRole('group', { name: /Image viewer/ })).toBeInTheDocument()
    })
  })
})

/**
 * Drives `useCropLayer` at a chosen transform.
 *
 * jsdom lays nothing out, so the real viewer always reports a zero-sized
 * container and would never ask for a crop. This hands the hook the numbers it
 * would have in a browser, and nothing else about it is stubbed.
 */
function CropProbe({ transform }: { transform: Transform }) {
  const layer = useCropLayer(JOB_ID, transform, PROBE_CONTAINER, OUTPUT)
  return <div data-testid="crop">{layer.url ?? 'none'}</div>
}
