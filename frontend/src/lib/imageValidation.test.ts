import { describe, expect, it, vi } from 'vitest'
import { sniffFormat, validateImageFile } from './imageValidation'
import type { UploadLimits } from '@/config/limits'

// Real magic-byte prefixes for each supported format.
const JPEG_HEADER = [0xff, 0xd8, 0xff, 0xe0]
const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const WEBP_HEADER = [
  0x52, 0x49, 0x46, 0x46, // "RIFF"
  0x24, 0x00, 0x00, 0x00, // chunk size
  0x57, 0x45, 0x42, 0x50, // "WEBP"
]
const GIF_HEADER = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]

function makeFile(
  header: readonly number[],
  {
    name = 'photo.png',
    type = 'image/png',
    padTo = 64,
  }: { name?: string; type?: string; padTo?: number } = {},
): File {
  const bytes = new Uint8Array(Math.max(padTo, header.length))
  bytes.set(header)
  return new File([bytes], name, { type })
}

const LIMITS: UploadLimits = {
  maxFileSizeBytes: 1024,
  maxInputPixels: 1_000_000,
  minDimension: 32,
  formats: ['JPEG', 'PNG', 'WEBP'],
}

const decodeAs = (width: number, height: number) => vi.fn(() => Promise.resolve({ width, height }))

describe('sniffFormat', () => {
  it('identifies the supported formats from their signatures', () => {
    expect(sniffFormat(new Uint8Array(JPEG_HEADER))).toBe('JPEG')
    expect(sniffFormat(new Uint8Array(PNG_HEADER))).toBe('PNG')
    expect(sniffFormat(new Uint8Array(WEBP_HEADER))).toBe('WEBP')
  })

  it('rejects an image format we do not support', () => {
    expect(sniffFormat(new Uint8Array(GIF_HEADER))).toBeNull()
  })

  it('rejects a RIFF container that is not WEBP', () => {
    const riffWave = [0x52, 0x49, 0x46, 0x46, 0, 0, 0, 0, 0x57, 0x41, 0x56, 0x45]
    expect(sniffFormat(new Uint8Array(riffWave))).toBeNull()
  })

  it('rejects arbitrary bytes', () => {
    expect(sniffFormat(new Uint8Array([0x4d, 0x5a, 0x90, 0x00]))).toBeNull()
  })
})

describe('validateImageFile', () => {
  it('accepts a valid image and reports metadata from its contents', async () => {
    const file = makeFile(PNG_HEADER, { name: 'shot.png' })

    const result = await validateImageFile(file, { limits: LIMITS, decode: decodeAs(800, 600) })

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.metadata).toEqual({
      name: 'shot.png',
      width: 800,
      height: 600,
      sizeBytes: file.size,
      format: 'PNG',
    })
  })

  it('determines the format from content, not the extension or MIME type', async () => {
    // A JPEG deliberately mislabelled as a PNG in both name and type.
    const file = makeFile(JPEG_HEADER, { name: 'actually-a-jpeg.png', type: 'image/png' })

    const result = await validateImageFile(file, { limits: LIMITS, decode: decodeAs(100, 100) })

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.metadata.format).toBe('JPEG')
  })

  it('rejects a non-image renamed to an image extension', async () => {
    const file = makeFile([0x4d, 0x5a, 0x90, 0x00], { name: 'payload.png', type: 'image/png' })

    const result = await validateImageFile(file, { limits: LIMITS, decode: decodeAs(10, 10) })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('unsupported_format')
    expect(result.problem.technical).toContain('declaredType=image/png')
  })

  it('does not decode a file whose header was rejected', async () => {
    const decode = decodeAs(10, 10)

    await validateImageFile(makeFile(GIF_HEADER), { limits: LIMITS, decode })

    expect(decode).not.toHaveBeenCalled()
  })

  it('rejects an empty file', async () => {
    const result = await validateImageFile(new File([], 'empty.png', { type: 'image/png' }), {
      limits: LIMITS,
      decode: decodeAs(10, 10),
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('corrupted_image')
    expect(result.problem.title).toBe('Empty file')
  })

  it('rejects a file above the size limit before reading its header', async () => {
    const decode = decodeAs(100, 100)
    const file = makeFile(PNG_HEADER, { padTo: 2048 })

    const result = await validateImageFile(file, { limits: LIMITS, decode })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('file_too_large')
    expect(result.problem.detail).toContain('2.0 KB')
    expect(decode).not.toHaveBeenCalled()
  })

  it('reports a decode failure as a corrupted image rather than throwing', async () => {
    const decode = vi.fn(() => Promise.reject(new Error('Image decode failed')))

    const result = await validateImageFile(makeFile(PNG_HEADER), { limits: LIMITS, decode })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('corrupted_image')
    expect(result.problem.detail).toContain('truncated or damaged')
    expect(result.problem.technical).toBe('Image decode failed')
  })

  it('rejects an image below the minimum dimension', async () => {
    const result = await validateImageFile(makeFile(PNG_HEADER), {
      limits: LIMITS,
      decode: decodeAs(16, 400),
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('image_too_small')
    expect(result.problem.technical).toContain('input=16x400')
  })

  it('rejects an image above the pixel limit', async () => {
    const result = await validateImageFile(makeFile(PNG_HEADER), {
      limits: LIMITS,
      decode: decodeAs(2000, 2000),
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    expect(result.problem.code).toBe('image_too_large')
    expect(result.problem.detail).toContain('4.0 MP')
  })

  it('accepts an image exactly at both boundaries', async () => {
    const result = await validateImageFile(makeFile(PNG_HEADER), {
      limits: LIMITS,
      decode: decodeAs(1000, 1000), // exactly maxInputPixels
    })

    expect(result.ok).toBe(true)
  })

  it('produces failures in the same shape as an API error', async () => {
    const result = await validateImageFile(makeFile(GIF_HEADER), {
      limits: LIMITS,
      decode: decodeAs(10, 10),
    })

    expect(result.ok).toBe(false)
    if (result.ok) return
    // Lets ErrorPanel render validation and server errors identically.
    expect(result.problem).toMatchObject({
      type: expect.stringContaining('unsupported_format'),
      title: expect.any(String),
      status: expect.any(Number),
      code: 'unsupported_format',
      detail: expect.any(String),
    })
  })
})
