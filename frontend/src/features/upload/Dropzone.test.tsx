import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { Dropzone } from './Dropzone'

function makeFile(name = 'photo.png'): File {
  return new File([new Uint8Array(8)], name, { type: 'image/png' })
}

function dropzoneOf(): HTMLElement {
  return screen.getByTestId('dropzone')
}

/**
 * fireEvent rather than dispatchEvent: it wraps the dispatch in act(), so the
 * resulting state update is flushed before assertions run.
 */
function drop(node: HTMLElement, files: File[]): void {
  fireEvent.drop(node, { dataTransfer: { files } })
}

describe('Dropzone', () => {
  it('states the instruction, formats and size limit', () => {
    render(<Dropzone onFileSelected={vi.fn()} />)

    expect(screen.getByText('Drop an image here')).toBeInTheDocument()
    expect(screen.getByText('or click to browse')).toBeInTheDocument()
    expect(screen.getByText(/JPG · JPEG · PNG · WEBP · up to 32 MB/)).toBeInTheDocument()
  })

  it('exposes a real file input that is reachable by keyboard', async () => {
    const user = userEvent.setup()
    render(<Dropzone onFileSelected={vi.fn()} />)

    const input = screen.getByLabelText(/Drop an image here/i)
    expect(input).toHaveAttribute('type', 'file')

    await user.tab()
    expect(input).toHaveFocus()
  })

  it('reports a file chosen through the picker', async () => {
    const onFileSelected = vi.fn()
    const user = userEvent.setup()
    render(<Dropzone onFileSelected={onFileSelected} />)

    const file = makeFile()
    await user.upload(screen.getByLabelText(/Drop an image here/i), file)

    expect(onFileSelected).toHaveBeenCalledWith(file)
  })

  it('reports a dropped file', () => {
    const onFileSelected = vi.fn()
    render(<Dropzone onFileSelected={onFileSelected} />)

    const file = makeFile('dropped.png')
    drop(dropzoneOf(), [file])

    expect(onFileSelected).toHaveBeenCalledWith(file)
  })

  it('ignores a drop that carries no file', () => {
    const onFileSelected = vi.fn()
    render(<Dropzone onFileSelected={onFileSelected} />)

    drop(dropzoneOf(), [])

    expect(onFileSelected).not.toHaveBeenCalled()
  })

  it('keeps the drag state through nested dragleave events', () => {
    render(<Dropzone onFileSelected={vi.fn()} />)
    const zone = dropzoneOf()

    const dragEnter = () => { fireEvent.dragEnter(zone) }
    const dragLeave = () => { fireEvent.dragLeave(zone) }

    // Entering a child fires enter again before the parent's leave.
    dragEnter()
    dragEnter()
    expect(zone.className).toContain('border-accent')

    dragLeave()
    // Still inside: one enter remains unmatched, so the highlight must persist.
    expect(zone.className).toContain('border-accent')

    dragLeave()
    expect(zone.className).not.toContain('border-accent')
  })

  it('accepts an image pasted from the clipboard', () => {
    const onFileSelected = vi.fn()
    render(<Dropzone onFileSelected={onFileSelected} />)

    const file = makeFile('clipboard.png')
    fireEvent.paste(window, { clipboardData: { files: [file] } })

    expect(onFileSelected).toHaveBeenCalledWith(file)
  })

  it('stops listening for pastes once unmounted', () => {
    const onFileSelected = vi.fn()
    const { unmount } = render(<Dropzone onFileSelected={onFileSelected} />)

    unmount()

    fireEvent.paste(window, { clipboardData: { files: [makeFile()] } })

    expect(onFileSelected).not.toHaveBeenCalled()
  })

  it('disables input and shows progress while a file is being read', () => {
    render(<Dropzone onFileSelected={vi.fn()} isLoading />)

    expect(screen.getByText('Reading image…')).toBeInTheDocument()
    expect(screen.getByLabelText(/Reading image/i)).toBeDisabled()
  })
})
