import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { App } from './App'
import { stubSystemApi } from '@/test/systemFixtures'

/**
 * Smoke tests for the composed application shell.
 *
 * These mount the real `App` — providers, router and all — so a mistake in how
 * the pieces are wired together fails here rather than only in a browser.
 */
describe('App', () => {
  beforeEach(() => {
    window.history.pushState({}, '', '/')
    // The shell mounts the status indicator on every route, and Settings
    // mounts the system panel, so all three endpoints have to answer.
    stubSystemApi()
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
      expect(screen.getByText('GPU acceleration enabled')).toBeInTheDocument()
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

  it('presents a working upload target and the real enhancement controls', () => {
    render(<App />)

    // The dropzone became real in Phase 4 and the settings rail in Phase 8.
    expect(screen.getByText('Drop an image here')).toBeInTheDocument()
    expect(screen.getByLabelText(/Drop an image here/i)).toHaveAttribute('type', 'file')
    expect(screen.getByRole('button', { name: 'Enhance' })).toBeInTheDocument()
  })

  it('still marks the regions that are not built', async () => {
    const user = userEvent.setup()
    render(<App />)

    // Settings still has unbuilt sections, and they say so rather than
    // looking functional. History became real in Phase 10.
    await user.click(screen.getByRole('link', { name: 'Settings' }))

    expect(screen.getAllByText(/Arrives in Phase/).length).toBeGreaterThan(0)
  })
})
