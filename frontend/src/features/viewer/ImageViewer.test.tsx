import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { resizeElement } from '@/test/browserStubs'
import type { LoadedImage } from '@/types/image'
import { ImageViewer } from './ImageViewer'

const CONTAINER = { width: 800, height: 600 }

const IMAGE: LoadedImage = {
  file: new File([new Uint8Array(8)], 'shot.png', { type: 'image/png' }),
  objectUrl: 'blob:pixelforge/test',
  metadata: { name: 'shot.png', width: 1600, height: 1200, sizeBytes: 2048, format: 'PNG' },
}

function renderViewer(image: LoadedImage = IMAGE) {
  const result = render(
    <TooltipProvider delayDuration={0}>
      <ImageViewer image={image} />
    </TooltipProvider>,
  )

  // jsdom reports every rect as zero. Deliver a real ResizeObserver entry so
  // the viewer sizes itself through exactly the path it uses in a browser.
  const viewport = screen.getByRole('group', { name: /Image viewer/ })
  act(() => { resizeElement(viewport, CONTAINER.width, CONTAINER.height) })
  return result
}

function transformOf(): string {
  return screen.getByRole('img').style.transform
}

describe('ImageViewer', () => {
  it('renders the image with descriptive alt text and intrinsic dimensions', () => {
    renderViewer()

    const img = screen.getByRole('img')
    expect(img).toHaveAccessibleName('Uploaded image: shot.png')
    expect(img).toHaveAttribute('width', '1600')
    expect(img).toHaveAttribute('height', '1200')
    expect(img).toHaveAttribute('src', 'blob:pixelforge/test')
  })

  it('exposes the zoom controls with accessible names', () => {
    renderViewer()

    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Fit to screen' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Actual size' })).toBeInTheDocument()
    expect(screen.getByLabelText('Zoom level')).toBeInTheDocument()
  })

  it('offers every documented zoom preset', () => {
    renderViewer()

    const options = screen
      .getAllByRole('option')
      .map((option) => (option as HTMLOptionElement).value)

    for (const preset of ['25', '50', '100', '200', '400']) {
      expect(options).toContain(preset)
    }
  })

  it('marks the view as fitted on first render', () => {
    renderViewer()

    expect(screen.getByRole('button', { name: 'Fit to screen' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  })

  it('switches to actual size and reports 100%', async () => {
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('button', { name: 'Actual size' }))

    expect(screen.getByLabelText('Zoom level')).toHaveValue('100')
    expect(transformOf()).toContain('scale(1)')
    expect(screen.getByRole('button', { name: 'Fit to screen' })).toHaveAttribute(
      'aria-pressed',
      'false',
    )
  })

  it('steps through presets with the zoom buttons', async () => {
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('button', { name: 'Actual size' }))
    await user.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(screen.getByLabelText('Zoom level')).toHaveValue('200')

    await user.click(screen.getByRole('button', { name: 'Zoom out' }))
    expect(screen.getByLabelText('Zoom level')).toHaveValue('100')
  })

  it('selects a zoom level from the dropdown', async () => {
    const user = userEvent.setup()
    renderViewer()

    await user.selectOptions(screen.getByLabelText('Zoom level'), '400')

    expect(transformOf()).toContain('scale(4)')
  })

  it('returns to the fitted view', async () => {
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('button', { name: 'Actual size' }))
    await user.click(screen.getByRole('button', { name: 'Fit to screen' }))

    // 1600x1200 inside 800x600 fits at exactly 50%.
    expect(screen.getByLabelText('Zoom level')).toHaveValue('50')
    expect(screen.getByRole('button', { name: 'Fit to screen' })).toHaveAttribute(
      'aria-pressed',
      'true',
    )
  })

  it('supports keyboard zoom shortcuts', async () => {
    const user = userEvent.setup()
    renderViewer()

    const viewport = screen.getByRole('group', { name: /Image viewer/ })
    viewport.focus()

    await user.keyboard('1')
    expect(screen.getByLabelText('Zoom level')).toHaveValue('100')

    await user.keyboard('+')
    expect(screen.getByLabelText('Zoom level')).toHaveValue('200')

    await user.keyboard('-')
    expect(screen.getByLabelText('Zoom level')).toHaveValue('100')

    await user.keyboard('0')
    expect(screen.getByLabelText('Zoom level')).toHaveValue('50')
  })

  it('names the keyboard shortcuts in the viewport label', () => {
    renderViewer()

    expect(
      screen.getByRole('group', { name: /0 fits to screen, 1 shows actual size/ }),
    ).toBeInTheDocument()
  })

  it('renders pixelated above 100% so pixels can be inspected', async () => {
    const user = userEvent.setup()
    renderViewer()

    await user.selectOptions(screen.getByLabelText('Zoom level'), '400')

    expect(screen.getByRole('img').style.imageRendering).toBe('pixelated')
  })

  it('smooths the image at or below 100%', async () => {
    const user = userEvent.setup()
    renderViewer()

    await user.click(screen.getByRole('button', { name: 'Actual size' }))

    expect(screen.getByRole('img').style.imageRendering).toBe('auto')
  })

  it('centres an image smaller than the viewport rather than stretching it', () => {
    renderViewer({
      ...IMAGE,
      metadata: { ...IMAGE.metadata, width: 400, height: 300 },
    })

    // Fit caps at 100%, so a small image stays sharp and is centred.
    expect(screen.getByLabelText('Zoom level')).toHaveValue('100')
    expect(transformOf()).toBe('translate(200px, 150px) scale(1)')
  })
})
