import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { TopNav } from './TopNav'
import { MOBILE_WIDTH, setViewportWidth } from '@/test/matchMedia'
import { jsonResponse, renderWithProviders } from '@/test/renderWithProviders'
import { useThemeStore } from '@/stores/useThemeStore'

const HEALTH_BODY = {
  status: 'ok',
  version: '0.1.0',
  environment: 'test',
  uptimeSeconds: 12,
}

describe('TopNav', () => {
  beforeEach(() => {
    useThemeStore.setState({ preference: 'dark', resolved: 'dark' })
    document.documentElement.classList.add('dark')
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.resolve(jsonResponse(HEALTH_BODY))),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders the three primary destinations on a desktop viewport', () => {
    renderWithProviders(<TopNav />)

    expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
    for (const label of ['Enhance', 'History', 'Settings']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
  })

  it('does not duplicate the Settings link when the main nav is visible', () => {
    renderWithProviders(<TopNav />)

    // A gear icon alongside the nav item would give two links the same
    // accessible name pointing at the same route.
    expect(screen.getAllByRole('link', { name: 'Settings' })).toHaveLength(1)
  })

  it('exposes an accessible theme toggle that flips the resolved theme', async () => {
    const user = userEvent.setup()
    renderWithProviders(<TopNav />)

    await user.click(screen.getByRole('button', { name: 'Switch to light theme' }))

    expect(useThemeStore.getState().resolved).toBe('light')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
    expect(screen.getByRole('button', { name: 'Switch to dark theme' })).toBeInTheDocument()
  })

  it('reports backend status from a real health request', async () => {
    renderWithProviders(<TopNav />)

    expect(screen.getByText('Checking backend')).toBeInTheDocument()
    await waitFor(() => {
      expect(screen.getByText('Backend online')).toBeInTheDocument()
    })
    expect(globalThis.fetch).toHaveBeenCalledWith(
      '/api/health',
      expect.objectContaining({ signal: expect.anything() }),
    )
  })

  it('reports the backend as offline when the request fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(() => Promise.reject(new TypeError('Failed to fetch'))),
    )
    renderWithProviders(<TopNav />)

    await waitFor(() => {
      expect(screen.getByText('Backend offline')).toBeInTheDocument()
    })
  })

  describe('compact viewport', () => {
    beforeEach(() => {
      setViewportWidth(MOBILE_WIDTH)
    })

    it('replaces the inline nav with a menu button and a settings shortcut', () => {
      renderWithProviders(<TopNav />)

      expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Open navigation menu' })).toBeInTheDocument()
      expect(screen.getAllByRole('link', { name: 'Settings' })).toHaveLength(1)
    })

    it('opens and closes the navigation menu', async () => {
      const user = userEvent.setup()
      renderWithProviders(<TopNav />)

      await user.click(screen.getByRole('button', { name: 'Open navigation menu' }))

      expect(screen.getByRole('navigation', { name: 'Main' })).toBeInTheDocument()
      const closeButton = screen.getByRole('button', { name: 'Close navigation menu' })
      expect(closeButton).toHaveAttribute('aria-expanded', 'true')

      await user.click(closeButton)

      expect(screen.queryByRole('navigation', { name: 'Main' })).not.toBeInTheDocument()
    })
  })
})
