import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it, vi } from 'vitest'
import { sniffFormat, validateImageFile } from './imageValidation'

/**
 * Validation against real encoder output.
 *
 * The other suite uses hand-written byte prefixes, which only proves the code
 * matches what I believed the signatures to be. These fixtures are produced by
 * Pillow (see the generator note below), so they carry whatever a real encoder
 * actually emits — including the JFIF APP0 segment in JPEG and the RIFF chunk
 * size in WEBP, neither of which the hand-written headers exercise.
 *
 * Regenerate with:
 *   backend/.venv/Scripts/python -c "from PIL import Image; ..."
 */
function fixture(name: string): Uint8Array<ArrayBuffer> {
  // Resolved from the project root (vitest's cwd) rather than import.meta.url,
  // which the jsdom environment does not report as a usable file URL.
  const contents = readFileSync(join(process.cwd(), 'src/test/fixtures', name))
  // Copied into a plain ArrayBuffer: a Buffer's backing store is typed as
  // ArrayBufferLike, which BlobPart does not accept.
  const bytes = new Uint8Array(contents.byteLength)
  bytes.set(contents)
  return bytes
}

function fixtureFile(name: string, declaredType: string): File {
  return new File([fixture(name)], name, { type: declaredType })
}

describe('sniffFormat against real encoder output', () => {
  it('identifies a real JPEG', () => {
    expect(sniffFormat(fixture('real.jpg'))).toBe('JPEG')
  })

  it('identifies a real PNG', () => {
    expect(sniffFormat(fixture('real.png'))).toBe('PNG')
  })

  it('identifies a real WEBP, whose signature is split across the RIFF header', () => {
    expect(sniffFormat(fixture('real.webp'))).toBe('WEBP')
  })

  it('rejects a real GIF, which is a genuine image we do not support', () => {
    expect(sniffFormat(fixture('real.gif'))).toBeNull()
  })

  it('still recognises a truncated PNG by its header', () => {
    // The header survives truncation, so the file passes the sniff and is
    // caught later by the decode step — which is exactly the intended split.
    expect(sniffFormat(fixture('truncated.png'))).toBe('PNG')
  })
})

describe('validateImageFile against real files', () => {
  const decode = vi.fn(() => Promise.resolve({ width: 64, height: 48 }))

  it('accepts a real JPEG mislabelled as a PNG, reporting the true format', async () => {
    const result = await validateImageFile(fixtureFile('real.jpg', 'image/png'), { decode })

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.metadata.format).toBe('JPEG')
  })

  it('accepts a real WEBP', async () => {
    const result = await validateImageFile(fixtureFile('real.webp', 'image/webp'), { decode })

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.metadata.format).toBe('WEBP')
  })

  it('rejects a real GIF as an unsupported format', async () => {
    const result = await validateImageFile(fixtureFile('real.gif', 'image/gif'), { decode })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('unsupported_format')
  })

  it('rejects a truncated PNG once decoding fails', async () => {
    // A real browser's createImageBitmap rejects here; the stub stands in for
    // that failure, since jsdom cannot decode images at all.
    const failingDecode = vi.fn(() => Promise.reject(new Error('Image is incomplete')))

    const result = await validateImageFile(fixtureFile('truncated.png', 'image/png'), {
      decode: failingDecode,
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('corrupted_image')
  })
})
