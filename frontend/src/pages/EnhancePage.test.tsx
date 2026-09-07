import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { EnhancePage } from './EnhancePage'
import { leakedObjectUrls, resizeElement } from '@/test/browserStubs'
import { renderWithProviders } from '@/test/renderWithProviders'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'

const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const GIF_HEADER = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]

function makeFile(header: readonly number[], name: string, byteLength = 1_887_437): File {
  const bytes = new Uint8Array(byteLength)
  bytes.set(header)
  return new File([bytes], name, { type: 'image/png' })
}

/**
 * Captured once, before any test replaces it. Reading `loadFile` off the store
 * inside the stub would capture an already-stubbed version on the second call,
 * nesting the stubs so the first test's decoder silently won.
 */
const REAL_LOAD_FILE = useWorkspaceStore.getState().loadFile

/**
 * jsdom cannot decode images, so the decoder is stubbed at the point where a
 * real browser would call `createImageBitmap`. Everything before it — magic
 * byte sniffing, size limits — and everything after runs for real.
 */
function stubDecode(width: number, height: number) {
  const decode = vi.fn(() => Promise.resolve({ width, height }))
  useWorkspaceStore.setState({ loadFile: (file) => REAL_LOAD_FILE(file, { decode }) })
  return decode
}

describe('EnhancePage', () => {
  beforeEach(() => {
    useWorkspaceStore.setState({
      source: null,
      problem: null,
      isLoading: false,
      loadFile: REAL_LOAD_FILE,
    })
  })

  it('shows the upload target and the four-step flow before an image is loaded', () => {
    renderWithProviders(<EnhancePage />)

    expect(screen.getByText('Drop an image here')).toBeInTheDocument()
    for (const step of ['Upload', 'Compare', 'Enhance', 'Download']) {
      expect(screen.getByText(step)).toBeInTheDocument()
    }
  })

  it('loads a real file and switches to the viewer with its metadata', async () => {
    stubDecode(1280, 720)
    const user = userEvent.setup()
    renderWithProviders(<EnhancePage />)

    await user.upload(
      screen.getByLabelText(/Drop an image here/i),
      makeFile(PNG_HEADER, 'holiday.png'),
    )

    await waitFor(() => {
      expect(screen.getByRole('img')).toBeInTheDocument()
    })

    expect(screen.getByRole('img')).toHaveAccessibleName('Uploaded image: holiday.png')
    // § 8 information panel, original side.
    expect(screen.getByText('1,280 × 720')).toBeInTheDocument()
    expect(screen.getByText('1.8 MB')).toBeInTheDocument()
    expect(screen.getByText('PNG')).toBeInTheDocument()
    expect(screen.getByText('0.9 MP')).toBeInTheDocument()
  })

  it('reports a rejected file without entering the viewer', async () => {
    stubDecode(1280, 720)
    const user = userEvent.setup()
    renderWithProviders(<EnhancePage />)

    await user.upload(
      screen.getByLabelText(/Drop an image here/i),
      makeFile(GIF_HEADER, 'animation.gif'),
    )

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument()
    })

    expect(screen.getByRole('alert')).toHaveTextContent('Unsupported file type')
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    // Still ready to accept another attempt.
    expect(screen.getByText('Drop an image here')).toBeInTheDocument()
  })

  it('keeps the technical detail behind the disclosure', async () => {
    stubDecode(1280, 720)
    const user = userEvent.setup()
    renderWithProviders(<EnhancePage />)

    await user.upload(
      screen.getByLabelText(/Drop an image here/i),
      makeFile(GIF_HEADER, 'animation.gif'),
    )
    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument()
    })

    expect(screen.queryByText(/declaredType=/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Technical details' }))
    expect(screen.getByText(/declaredType=image\/png/)).toBeInTheDocument()
  })

  it('removes the image and revokes its object URL', async () => {
    stubDecode(1280, 720)
    const user = userEvent.setup()
    renderWithProviders(<EnhancePage />)

    await user.upload(
      screen.getByLabelText(/Drop an image here/i),
      makeFile(PNG_HEADER, 'holiday.png'),
    )
    await waitFor(() => {
      expect(screen.getByRole('img')).toBeInTheDocument()
    })

    await user.click(screen.getByRole('button', { name: /Remove/ }))

    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(screen.getByText('Drop an image here')).toBeInTheDocument()
    expect(leakedObjectUrls()).toEqual([])
  })

  it('offers zoom controls once an image is loaded', async () => {
    stubDecode(1600, 1200)
    const user = userEvent.setup()
    renderWithProviders(<EnhancePage />)

    await user.upload(
      screen.getByLabelText(/Drop an image here/i),
      makeFile(PNG_HEADER, 'wide.png'),
    )
    await waitFor(() => {
      expect(screen.getByRole('img')).toBeInTheDocument()
    })

    const viewport = screen.getByRole('group', { name: /Image viewer/ })
    act(() => { resizeElement(viewport, 800, 600) })

    expect(screen.getByLabelText('Zoom level')).toHaveValue('50')
    await user.click(screen.getByRole('button', { name: 'Actual size' }))
    expect(screen.getByLabelText('Zoom level')).toHaveValue('100')
  })

  it('does not claim the result panels are working yet', async () => {
    stubDecode(1280, 720)
    const user = userEvent.setup()
    renderWithProviders(<EnhancePage />)

    await user.upload(
      screen.getByLabelText(/Drop an image here/i),
      makeFile(PNG_HEADER, 'holiday.png'),
    )
    await waitFor(() => {
      expect(screen.getByRole('img')).toBeInTheDocument()
    })

    // The "Enhanced" side of the info panel must state its phase, not show
    // fabricated output numbers.
    expect(screen.getByText('Enhanced')).toBeInTheDocument()
    expect(screen.getAllByText(/Arrives in Phase/).length).toBeGreaterThan(0)
  })
})
