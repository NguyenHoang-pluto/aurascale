import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { jsonResponse } from '@/test/renderWithProviders'

/**
 * Smoke tests for the composed application shell.
 *
 * These mount the real `App` — providers, router and all — so a mistake in how
 * the pieces are wired together fails here rather than only in a browser.
 */
describe('App', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/')
    vi.stubGlobal(
      'fetch',
      vi.fn(() =>
        Promise.resolve(
          jsonResponse({
            status: 'ok',
            version: '0.1.0',
            environment: 'test',
            uptimeSeconds: 1,
          }),
        ),
      ),
    )
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('mounts and renders the enhance workspace at the root route', async () => {
    render(<App />)

    expect(screen.getByRole('banner')).toBeInTheDocument()
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: 'Image workspace' })).toBeInTheDocument()
    expect(
      screen.getByRole('complementary', { name: 'Enhancement settings' }),
    ).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText('Backend online')).toBeInTheDocument()
    })
  })

  it('offers a skip link as the first focusable element', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.tab()

    const skipLink = screen.getByRole('link', { name: 'Skip to content' })
    expect(skipLink).toHaveFocus()
    expect(skipLink).toHaveAttribute('href', '#main')
  })

  it('navigates between the primary screens', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('link', { name: 'History' }))
    expect(screen.getByRole('heading', { level: 1, name: 'History' })).toBeInTheDocument()

    await user.click(screen.getByRole('link', { name: 'Settings' }))
    expect(screen.getByRole('heading', { level: 1, name: 'Settings' })).toBeInTheDocument()
  })

  it('shows a not-found screen for an unknown route', () => {
    window.history.pushState({}, '', '/no-such-page')
    render(<App />)

    expect(screen.getByText('Page not found')).toBeInTheDocument()
  })

  it('states which phase implements each unbuilt region instead of faking it', () => {
    render(<App />)

    // The workspace must never present a dropzone that does nothing.
    expect(screen.queryByText(/Drop an image here/i)).not.toBeInTheDocument()
    expect(screen.getAllByText(/Arrives in Phase/).length).toBeGreaterThan(0)
  })
})
