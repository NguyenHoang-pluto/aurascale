import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { HistoryPage } from '@/pages/HistoryPage'
import {
  describeSizes,
  formatTimestamp,
  hasResult,
  isRunning,
  pageCount,
  pageRange,
  statusLabel,
} from './historyPresentation'
import { jsonResponse, renderWithProviders } from '@/test/renderWithProviders'
import { GPU_SYSTEM, HEALTH } from '@/test/systemFixtures'
import type { JobRecord } from '@/types/job'

/**
 * The history screen.
 *
 * The backend is stubbed at `fetch`; the real API client, the real query
 * hooks and the real components run.
 */

function job(index: number, overrides: Partial<JobRecord> = {}): JobRecord {
  const id = String(index).padStart(32, '0')
  return {
    jobId: id,
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

interface BackendOptions {
  items?: JobRecord[]
  total?: number
  fail?: boolean
  thumbnailFails?: boolean
}

function stubBackend(options: BackendOptions = {}) {
  const items = options.items ?? [job(1)]
  const requests: string[] = []
  const deletions: string[] = []

  const fetchMock = vi.fn((url: string, init?: RequestInit) => {
    const method = init?.method ?? 'GET'

    if (url.includes('/api/system')) return Promise.resolve(jsonResponse(GPU_SYSTEM))
    if (url.includes('/api/health')) return Promise.resolve(jsonResponse(HEALTH))

    if (url.includes('/thumbnail')) {
      if (options.thumbnailFails === true) {
        return Promise.resolve(new Response(null, { status: 404 }))
      }
      return Promise.resolve(new Response(new Blob([new Uint8Array([1])])))
    }

    if (method === 'DELETE') {
      deletions.push(url)
      return Promise.resolve(new Response(null, { status: 204 }))
    }

    if (url.includes('/api/jobs')) {
      requests.push(url)
      if (options.fail === true) {
        return Promise.resolve(
          jsonResponse(
            {
              type: 'https://pixelforge.ai/errors/backend_unavailable',
              title: 'Cannot reach the backend',
              status: 503,
              code: 'backend_unavailable',
              detail: 'The PixelForge backend is not responding.',
            },
            503,
          ),
        )
      }

      // Serve the slice the query string asks for, so paging is exercised
      // rather than assumed.
      const parsed = new URL(url, 'http://test')
      const limit = Number(parsed.searchParams.get('limit') ?? 20)
      const offset = Number(parsed.searchParams.get('offset') ?? 0)
      const status = parsed.searchParams.get('status')
      const filtered = status === null ? items : items.filter((item) => item.status === status)

      return Promise.resolve(
        jsonResponse({
          items: filtered.slice(offset, offset + limit),
          total: options.total ?? filtered.length,
          limit,
          offset,
        }),
      )
    }

    return Promise.reject(new TypeError('Failed to fetch'))
  })

  vi.stubGlobal('fetch', fetchMock)
  return { requests, deletions }
}

beforeEach(() => {
  // Each test installs its own backend; start from a clean global.
  vi.unstubAllGlobals()
})

afterEach(() => {
  vi.unstubAllGlobals()
})

// ------------------------------------------------------------------- states

describe('history states', () => {
  it('shows a loading state before the first page arrives', () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)

    expect(screen.getByLabelText('Loading history')).toBeInTheDocument()
  })

  it('says plainly when there is nothing yet', async () => {
    stubBackend({ items: [] })
    renderWithProviders(<HistoryPage />)

    await waitFor(() => {
      expect(screen.getByText('Nothing here yet')).toBeInTheDocument()
    })
    expect(screen.getByText(/Enhance an image and it will appear here/)).toBeInTheDocument()
  })

  it('reports an unreachable backend with a retry', async () => {
    stubBackend({ fail: true })
    renderWithProviders(<HistoryPage />)

    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Could not load history')
    })
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('retries when asked', async () => {
    const { requests } = stubBackend({ fail: true })
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
    })
    const before = requests.length

    await user.click(screen.getByRole('button', { name: 'Try again' }))

    await waitFor(() => {
      expect(requests.length).toBeGreaterThan(before)
    })
  })
})

// -------------------------------------------------------------------- entry

describe('a history entry', () => {
  it('shows what the job did', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)

    const card = await screen.findByRole('article')
    const entry = within(card)

    expect(entry.getByText('Completed')).toBeInTheDocument()
    expect(entry.getByText('realesr-general-x4v3')).toBeInTheDocument()
    expect(entry.getByText('4x')).toBeInTheDocument()
    expect(entry.getByText('320 × 240 → 1,280 × 960')).toBeInTheDocument()
    expect(entry.getByText('8.42s')).toBeInTheDocument()
  })

  it('loads its thumbnail from the endpoint, not the full result', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)

    const image = await screen.findByRole('img')
    expect(image).toHaveAttribute('src', expect.stringContaining('/thumbnail'))
    expect(image.getAttribute('src')).not.toContain('/result')
    expect(image).toHaveAttribute('loading', 'lazy')
  })

  it('says so rather than showing a broken image when the tile is missing', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)
    const image = await screen.findByRole('img')

    image.dispatchEvent(new Event('error'))

    await waitFor(() => {
      expect(screen.getByText('Preview unavailable')).toBeInTheDocument()
    })
  })

  it('offers no result for a failed job', async () => {
    stubBackend({
      items: [
        job(1, {
          status: 'failed',
          output: null,
          error: { code: 'out_of_memory', detail: 'Not enough memory to enhance this image.' },
        }),
      ],
    })
    renderWithProviders(<HistoryPage />)

    const entry = within(await screen.findByRole('article'))
    expect(entry.getByText('Failed')).toBeInTheDocument()
    expect(entry.getByText('No result')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'View' })).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Download/ })).not.toBeInTheDocument()
    // The reason is shown, not swallowed.
    expect(entry.getByText(/Not enough memory/)).toBeInTheDocument()
  })

  it('offers no result for a cancelled job', async () => {
    stubBackend({ items: [job(1, { status: 'cancelled', output: null, progress: 0 })] })
    renderWithProviders(<HistoryPage />)

    const entry = within(await screen.findByRole('article'))
    expect(entry.getByText('Cancelled')).toBeInTheDocument()
    expect(screen.queryByRole('link', { name: /Download/ })).not.toBeInTheDocument()
  })

  it('does not offer delete for a job that is still running', async () => {
    // DELETE on a running job cancels it, which is a different action that
    // belongs in the workspace.
    stubBackend({ items: [job(1, { status: 'processing', output: null, progress: 40 })] })
    renderWithProviders(<HistoryPage />)

    const entry = within(await screen.findByRole('article'))
    expect(entry.getByText('In progress')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /Delete/ })).not.toBeInTheDocument()
  })

  it('links the download straight at the result endpoint', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)

    const link = await screen.findByRole('link', { name: /Download/ })
    expect(link).toHaveAttribute('href', expect.stringContaining('/result'))
    expect(link).toHaveAttribute('download')
  })
})

// ------------------------------------------------------------------ actions

describe('viewing a result', () => {
  it('opens the result on its own', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')

    await user.click(screen.getByRole('button', { name: 'View' }))

    const viewer = await screen.findByRole('region', { name: 'Result viewer' })
    // The preview, not the full file.
    expect(within(viewer).getByAltText('Enhanced result')).toHaveAttribute(
      'src',
      expect.stringContaining('/preview'),
    )
  })

  it('offers no before/after for a past job', async () => {
    // The original is not kept, so there is nothing honest to compare against.
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')

    await user.click(screen.getByRole('button', { name: 'View' }))

    await screen.findByRole('region', { name: 'Result viewer' })
    expect(screen.queryByRole('radiogroup', { name: 'Comparison mode' })).not.toBeInTheDocument()
  })

  it('closes again', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')
    await user.click(screen.getByRole('button', { name: 'View' }))
    await screen.findByRole('region', { name: 'Result viewer' })

    await user.click(screen.getByRole('button', { name: /Close/ }))

    expect(screen.queryByRole('region', { name: 'Result viewer' })).not.toBeInTheDocument()
  })
})

describe('deleting an entry', () => {
  it('asks before removing anything', async () => {
    const { deletions } = stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')

    await user.click(screen.getByRole('button', { name: /Delete the job/ }))

    expect(screen.getByRole('alertdialog', { name: 'Confirm deletion' })).toBeInTheDocument()
    expect(deletions).toEqual([])
  })

  it('does nothing when the confirmation is dismissed', async () => {
    const { deletions } = stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')
    await user.click(screen.getByRole('button', { name: /Delete the job/ }))

    await user.click(screen.getByRole('button', { name: 'Keep' }))

    expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument()
    expect(deletions).toEqual([])
  })

  it('deletes through the existing endpoint and refetches', async () => {
    const { deletions, requests } = stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')
    const before = requests.length

    await user.click(screen.getByRole('button', { name: /Delete the job/ }))
    await user.click(screen.getByRole('button', { name: 'Delete' }))

    await waitFor(() => {
      expect(deletions).toHaveLength(1)
    })
    expect(deletions[0]).toContain('/api/jobs/')
    await waitFor(() => {
      expect(requests.length).toBeGreaterThan(before)
    })
  })
})

// ------------------------------------------------------- paging and filters

describe('paging', () => {
  it('shows no pager when everything fits on one page', async () => {
    stubBackend({ items: [job(1), job(2)] })
    renderWithProviders(<HistoryPage />)
    await screen.findAllByRole('article')

    expect(screen.queryByRole('navigation', { name: 'History pages' })).not.toBeInTheDocument()
  })

  it('pages through a longer history', async () => {
    const items = Array.from({ length: 25 }, (_, index) => job(index + 1))
    const { requests } = stubBackend({ items })
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findAllByRole('article')

    expect(screen.getByText('Showing 1–20 of 25')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Next' }))

    await waitFor(() => {
      expect(requests.some((url) => url.includes('offset=20'))).toBe(true)
    })
    await waitFor(() => {
      expect(screen.getByText('Page 2 of 2')).toBeInTheDocument()
    })
  })

  it('disables the edges of the pager', async () => {
    stubBackend({ items: Array.from({ length: 25 }, (_, index) => job(index + 1)) })
    renderWithProviders(<HistoryPage />)
    await screen.findAllByRole('article')

    expect(screen.getByRole('button', { name: 'Previous' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Next' })).toBeEnabled()
  })
})

describe('filtering', () => {
  it('asks the backend for one status', async () => {
    const { requests } = stubBackend({
      items: [job(1), job(2, { status: 'failed', output: null })],
    })
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findAllByRole('article')

    await user.click(screen.getByRole('radio', { name: 'Failed' }))

    await waitFor(() => {
      expect(requests.some((url) => url.includes('status=failed'))).toBe(true)
    })
  })

  it('returns to the first page when the filter changes', async () => {
    const items = Array.from({ length: 25 }, (_, index) => job(index + 1))
    const { requests } = stubBackend({ items })
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findAllByRole('article')
    await user.click(screen.getByRole('button', { name: 'Next' }))
    await waitFor(() => {
      expect(screen.getByText('Page 2 of 2')).toBeInTheDocument()
    })

    await user.click(screen.getByRole('radio', { name: 'Completed' }))

    // An offset from the previous filter would point at the wrong slice.
    await waitFor(() => {
      expect(
        requests.some((url) => url.includes('status=completed') && url.includes('offset=0')),
      ).toBe(true)
    })
  })

  it('says when a filter matches nothing', async () => {
    stubBackend({ items: [job(1)] })
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')

    await user.click(screen.getByRole('radio', { name: 'Failed' }))

    await waitFor(() => {
      expect(screen.getByText('Nothing matches that filter')).toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------- retention

describe('retention', () => {
  it('states the 24-hour window, so nobody treats this as an archive', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)

    expect(screen.getByText(/removed 24 hours after they finish/)).toBeInTheDocument()
    expect(screen.getByText(/download anything you want to keep/i)).toBeInTheDocument()
  })
})

// ------------------------------------------------------------ accessibility

describe('accessibility', () => {
  it('labels the grid controls', async () => {
    stubBackend({ items: Array.from({ length: 25 }, (_, index) => job(index + 1)) })
    renderWithProviders(<HistoryPage />)
    await screen.findAllByRole('article')

    expect(screen.getByRole('radiogroup', { name: 'Filter by status' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'History pages' })).toBeInTheDocument()
  })

  it('gives every entry an accessible name', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)

    const card = await screen.findByRole('article')
    expect(card).toHaveAccessibleName(/Job from/)
  })

  it('names the delete button after the job it deletes', async () => {
    stubBackend()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')

    expect(screen.getByRole('button', { name: /Delete the job from/ })).toBeInTheDocument()
  })

  it('can be driven from the keyboard alone', async () => {
    stubBackend()
    const user = userEvent.setup()
    renderWithProviders(<HistoryPage />)
    await screen.findByRole('article')

    // Tab until the view action has focus, then activate it.
    const view = screen.getByRole('button', { name: 'View' })
    view.focus()
    await user.keyboard('{Enter}')

    expect(await screen.findByRole('region', { name: 'Result viewer' })).toBeInTheDocument()
  })
})

// ----------------------------------------------------------- pure behaviour

describe('presentation helpers', () => {
  it('describes the size change in one line', () => {
    expect(describeSizes(job(1))).toBe('320 × 240 → 1,280 × 960')
  })

  it('shows only the input when there is no output', () => {
    expect(describeSizes(job(1, { output: null }))).toBe('320 × 240')
  })

  it('knows which jobs have a file behind them', () => {
    expect(hasResult(job(1))).toBe(true)
    expect(hasResult(job(1, { status: 'failed', output: null }))).toBe(false)
    // A completed job whose output was swept has nothing to offer either.
    expect(hasResult(job(1, { output: null }))).toBe(false)
  })

  it('knows which jobs are still moving', () => {
    expect(isRunning(job(1, { status: 'queued' }))).toBe(true)
    expect(isRunning(job(1, { status: 'processing' }))).toBe(true)
    expect(isRunning(job(1))).toBe(false)
  })

  it('labels every status', () => {
    for (const status of ['queued', 'processing', 'completed', 'failed', 'cancelled'] as const) {
      expect(statusLabel(status)).not.toBe('')
    }
  })

  it('counts pages from a total', () => {
    expect(pageCount(0, 20)).toBe(1)
    expect(pageCount(20, 20)).toBe(1)
    expect(pageCount(21, 20)).toBe(2)
    expect(pageCount(41, 20)).toBe(3)
  })

  it('describes the range a page covers', () => {
    expect(pageRange(0, 20, 42)).toEqual({ first: 1, last: 20 })
    expect(pageRange(2, 20, 42)).toEqual({ first: 41, last: 42 })
    expect(pageRange(0, 20, 0)).toEqual({ first: 0, last: 0 })
  })

  it('shows a time today as today', () => {
    // Built from one instant so the comparison is timezone-independent: a UTC
    // literal can fall on either side of local midnight.
    const now = new Date()
    const earlier = new Date(now.getTime() - 60 * 60 * 1000)

    expect(formatTimestamp(earlier.toISOString(), now)).toMatch(/^Today /)
  })

  it('dates anything older', () => {
    const now = new Date()
    const lastWeek = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)

    expect(formatTimestamp(lastWeek.toISOString(), now)).not.toMatch(/^Today /)
  })

  it('does not pretend to read a broken timestamp', () => {
    expect(formatTimestamp('not a date')).toBe('Unknown')
  })
})
