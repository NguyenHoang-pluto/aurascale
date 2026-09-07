import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it } from 'vitest'
import { TopNav } from './TopNav'
import { useThemeStore } from '@/stores/useThemeStore'

function renderNav(initialPath = '/') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <TopNav />
    </MemoryRouter>,
  )
}

describe('TopNav', () => {
  beforeEach(() => {
    useThemeStore.setState({ theme: 'dark' })
  })

  it('renders the three primary destinations', () => {
    renderNav()
    const nav = screen.getByRole('navigation', { name: 'Main' })
    expect(nav).toBeInTheDocument()
    for (const label of ['Enhance', 'History', 'Settings']) {
      expect(screen.getByRole('link', { name: label })).toBeInTheDocument()
    }
  })

  it('exposes an accessible theme toggle that flips the stored theme', async () => {
    const user = userEvent.setup()
    renderNav()

    const toggle = screen.getByRole('button', { name: 'Switch to light theme' })
    await user.click(toggle)

    expect(useThemeStore.getState().theme).toBe('light')
    expect(document.documentElement.classList.contains('dark')).toBe(false)
  })
})
