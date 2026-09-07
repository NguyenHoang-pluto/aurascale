import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { useState } from 'react'
import { Button } from './button'
import { Field } from './field'
import { Progress } from './progress'
import { SegmentedControl } from './segmented-control'
import { StatusIndicator } from './status'
import { Switch } from './switch'

describe('Button', () => {
  it('defaults to type="button" so it never submits a form by accident', () => {
    render(<Button>Go</Button>)

    expect(screen.getByRole('button', { name: 'Go' })).toHaveAttribute('type', 'button')
  })

  it('honours an explicit type', () => {
    render(<Button type="submit">Send</Button>)

    expect(screen.getByRole('button', { name: 'Send' })).toHaveAttribute('type', 'submit')
  })

  it('renders as its child when asChild is set, keeping the link role', () => {
    render(
      <Button asChild>
        <a href="/history">History</a>
      </Button>,
    )

    const link = screen.getByRole('link', { name: 'History' })
    expect(link).toBeInTheDocument()
    expect(link).not.toHaveAttribute('type')
  })

  it('does not fire when disabled', async () => {
    const onClick = vi.fn()
    const user = userEvent.setup()
    render(
      <Button disabled onClick={onClick}>
        Enhance
      </Button>,
    )

    await user.click(screen.getByRole('button', { name: 'Enhance' }))

    expect(onClick).not.toHaveBeenCalled()
  })
})

describe('StatusIndicator', () => {
  it('always renders a text label, never colour alone', () => {
    render(<StatusIndicator tone="danger" label="Backend offline" />)

    expect(screen.getByText('Backend offline')).toBeInTheDocument()
  })

  it('marks itself as a live region when asked', () => {
    render(<StatusIndicator live tone="pending" label="Running inference" />)

    const status = screen.getByRole('status')
    expect(status).toHaveAttribute('aria-live', 'polite')
    expect(status).toHaveTextContent('Running inference')
  })
})

describe('SegmentedControl', () => {
  function Harness() {
    const [value, setValue] = useState<'2' | '4' | '8'>('4')
    return (
      <SegmentedControl
        name="scale"
        label="Upscale factor"
        value={value}
        onChange={setValue}
        options={[
          { value: '2', label: '2x' },
          { value: '4', label: '4x' },
          { value: '8', label: '8x' },
        ]}
      />
    )
  }

  it('exposes a named radiogroup with the current selection', () => {
    render(<Harness />)

    expect(screen.getByRole('radiogroup', { name: 'Upscale factor' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: '4x' })).toBeChecked()
  })

  it('changes selection on click', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    await user.click(screen.getByRole('radio', { name: '8x' }))

    expect(screen.getByRole('radio', { name: '8x' })).toBeChecked()
    expect(screen.getByRole('radio', { name: '4x' })).not.toBeChecked()
  })

  it('supports arrow-key navigation, which native radios provide', async () => {
    const user = userEvent.setup()
    render(<Harness />)

    await user.click(screen.getByRole('radio', { name: '4x' }))
    await user.keyboard('{ArrowRight}')

    expect(screen.getByRole('radio', { name: '8x' })).toBeChecked()
  })

  it('does not select a disabled option', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    render(
      <SegmentedControl
        name="mode"
        label="Comparison mode"
        value="slider"
        onChange={onChange}
        options={[
          { value: 'slider', label: 'Slider' },
          { value: 'split', label: 'Split', disabled: true },
        ]}
      />,
    )

    await user.click(screen.getByRole('radio', { name: 'Split' }))

    expect(onChange).not.toHaveBeenCalled()
  })
})

describe('Field', () => {
  it('links its label and description to the control', () => {
    render(
      <Field label="Preserve metadata" description="Keep EXIF and ICC profile.">
        {({ id, describedBy }) => (
          <Switch id={id} aria-describedby={describedBy} checked={false} />
        )}
      </Field>,
    )

    const control = screen.getByRole('switch', { name: 'Preserve metadata' })
    expect(control).toHaveAccessibleDescription('Keep EXIF and ICC profile.')
  })

  it('marks an unimplemented control as coming soon rather than wiring it to nothing', () => {
    render(
      <Field label="Artifact reduction" comingSoon orientation="horizontal">
        {({ id }) => <Switch id={id} checked={false} disabled />}
      </Field>,
    )

    expect(screen.getByText('Coming soon')).toBeInTheDocument()
    expect(screen.getByRole('switch', { name: /Artifact reduction/ })).toBeDisabled()
  })
})

describe('Progress', () => {
  it('exposes an accessible name and the current value', () => {
    render(<Progress value={62} label="Enhancement progress" />)

    const bar = screen.getByRole('progressbar', { name: 'Enhancement progress' })
    expect(bar).toHaveAttribute('aria-valuenow', '62')
  })

  it('clamps out-of-range and non-finite values', () => {
    const { rerender } = render(<Progress value={140} label="p" />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '100')

    rerender(<Progress value={Number.NaN} label="p" />)
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0')
  })
})
