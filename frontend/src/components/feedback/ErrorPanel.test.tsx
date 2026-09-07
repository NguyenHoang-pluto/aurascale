import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { ErrorPanel } from './ErrorPanel'

const TECHNICAL =
  'RuntimeError: CUDA out of memory. Tried to allocate 1.24 GiB (GPU 0; 4.00 GiB total capacity)'

describe('ErrorPanel', () => {
  it('leads with the human-readable message, not the traceback', () => {
    render(
      <ErrorPanel
        title="Not enough memory"
        detail="Your image is too large to process with the available GPU memory."
        technical={TECHNICAL}
        code="out_of_memory"
      />,
    )

    const alert = screen.getByRole('alert')
    expect(alert).toHaveTextContent('Not enough memory')
    expect(alert).toHaveTextContent('too large to process')
    // Technical detail exists but is collapsed, so it is not shown initially.
    expect(screen.queryByText(TECHNICAL)).not.toBeInTheDocument()
  })

  it('reveals technical details on demand', async () => {
    const user = userEvent.setup()
    render(
      <ErrorPanel
        title="Not enough memory"
        detail="Your image is too large."
        technical={TECHNICAL}
        code="out_of_memory"
      />,
    )

    await user.click(screen.getByRole('button', { name: 'Technical details' }))

    expect(screen.getByText(TECHNICAL)).toBeInTheDocument()
    expect(screen.getByText('code: out_of_memory')).toBeInTheDocument()
  })

  it('omits the disclosure entirely when there is nothing technical to show', () => {
    render(<ErrorPanel title="Upload failed" detail="Please try again." />)

    expect(
      screen.queryByRole('button', { name: 'Technical details' }),
    ).not.toBeInTheDocument()
  })

  it('offers a retry action only when a handler is supplied', async () => {
    const onRetry = vi.fn()
    const user = userEvent.setup()
    const { rerender } = render(
      <ErrorPanel title="Failed" detail="Network error." onRetry={onRetry} />,
    )

    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(onRetry).toHaveBeenCalledOnce()

    rerender(<ErrorPanel title="Failed" detail="Network error." />)
    expect(screen.queryByRole('button', { name: 'Try again' })).not.toBeInTheDocument()
  })
})
