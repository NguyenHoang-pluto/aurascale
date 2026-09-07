import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { BackendStatusIndicator } from './BackendStatusIndicator'
import { describeStatus } from './describeStatus'
import { SystemPanel } from './SystemPanel'
import { renderWithProviders } from '@/test/renderWithProviders'
import { CPU_SYSTEM, GPU_SYSTEM, HEALTH, MODELS, stubApi } from '@/test/systemFixtures'

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('describeStatus', () => {
  it('prioritises reachability over any device claim', () => {
    // Claiming "CPU mode" while the server is down would be a guess.
    expect(describeStatus('offline', GPU_SYSTEM)).toEqual({
      tone: 'danger',
      label: 'Backend offline',
    })
  })

  it('waits for the system report before naming a device', () => {
    expect(describeStatus('online', undefined)).toEqual({
      tone: 'pending',
      label: 'Reading capabilities',
    })
  })

  it('reports GPU acceleration only when the server confirmed a CUDA device', () => {
    expect(describeStatus('online', GPU_SYSTEM)).toEqual({
      tone: 'success',
      label: 'GPU acceleration enabled',
    })
  })

  it('reports CPU mode as a warning, not a failure', () => {
    expect(describeStatus('online', CPU_SYSTEM)).toEqual({
      tone: 'warning',
      label: 'CPU mode',
    })
  })
})

describe('BackendStatusIndicator', () => {
  beforeEach(() => {
    stubApi({ '/api/health': HEALTH, '/api/system': GPU_SYSTEM })
  })

  it('announces GPU acceleration once the system report arrives', async () => {
    renderWithProviders(<BackendStatusIndicator />)

    await waitFor(() => {
      expect(screen.getByText('GPU acceleration enabled')).toBeInTheDocument()
    })
    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite')
  })

  it('announces CPU mode when no GPU is present', async () => {
    stubApi({ '/api/health': HEALTH, '/api/system': CPU_SYSTEM })
    renderWithProviders(<BackendStatusIndicator />)

    await waitFor(() => {
      expect(screen.getByText('CPU mode')).toBeInTheDocument()
    })
  })

  it('reports the backend as offline rather than guessing at a device', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )
    renderWithProviders(<BackendStatusIndicator />)

    await waitFor(() => {
      expect(screen.getByText('Backend offline')).toBeInTheDocument()
    })
  })
})

describe('SystemPanel', () => {
  it('renders the measured GPU, CUDA and host details', async () => {
    stubApi({ '/api/system': GPU_SYSTEM, '/api/models': MODELS })
    renderWithProviders(<SystemPanel />)

    await waitFor(() => {
      expect(screen.getByText('NVIDIA GeForce RTX 3050 Laptop GPU')).toBeInTheDocument()
    })

    expect(screen.getByText('3.3 GB free / 4.0 GB')).toBeInTheDocument()
    expect(screen.getByText('2.7.1+cu118')).toBeInTheDocument()
    expect(screen.getByText('11.8')).toBeInTheDocument()
    expect(screen.getByText('AMD Ryzen 5 5625U with Radeon Graphics')).toBeInTheDocument()
    expect(screen.getByText('6 physical / 12 logical')).toBeInTheDocument()
    expect(screen.getByText('Enabled')).toBeInTheDocument()
  })

  it('presents a CPU-only machine as supported, not broken', async () => {
    stubApi({ '/api/system': CPU_SYSTEM, '/api/models': MODELS })
    renderWithProviders(<SystemPanel />)

    await waitFor(() => {
      expect(screen.getByText('CPU mode')).toBeInTheDocument()
    })

    expect(screen.getByText('None detected')).toBeInTheDocument()
    expect(screen.getByText('No')).toBeInTheDocument()
    expect(screen.getByText('Disabled')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('shows which model weights are present on disk', async () => {
    stubApi({ '/api/system': GPU_SYSTEM, '/api/models': MODELS })
    renderWithProviders(<SystemPanel />)

    await waitFor(() => {
      expect(screen.getByText('Real-ESRGAN x4 Plus')).toBeInTheDocument()
    })

    expect(screen.getByText('Not downloaded')).toBeInTheDocument()
    expect(screen.getByText('4.7 MB')).toBeInTheDocument()
    expect(screen.getByText(/RRDBNet · 4x/)).toBeInTheDocument()
    expect(screen.getByText(/SRVGGNetCompact · 4x · denoise/)).toBeInTheDocument()
  })

  it('surfaces an unreachable backend with a retry rather than a blank panel', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )
    renderWithProviders(<SystemPanel />)

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument()
    })

    expect(screen.getByRole('alert')).toHaveTextContent('Cannot reach the backend')
    expect(screen.getByRole('button', { name: 'Try again' })).toBeInTheDocument()
  })

  it('refetches on demand', async () => {
    const fetchMock = stubApi({ '/api/system': GPU_SYSTEM, '/api/models': MODELS })
    const user = userEvent.setup()
    renderWithProviders(<SystemPanel />)

    await waitFor(() => {
      expect(screen.getByText('NVIDIA GeForce RTX 3050 Laptop GPU')).toBeInTheDocument()
    })
    const before = fetchMock.mock.calls.filter(([url]) =>
      String(url).includes('/api/system'),
    ).length

    await user.click(screen.getByRole('button', { name: 'Refresh system status' }))

    await waitFor(() => {
      const after = fetchMock.mock.calls.filter(([url]) =>
        String(url).includes('/api/system'),
      ).length
      expect(after).toBeGreaterThan(before)
    })
  })
})
