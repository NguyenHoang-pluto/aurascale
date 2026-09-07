import { describe, expect, it } from 'vitest'
import { formatBytes, formatDimensions, formatDuration, formatMegapixels } from './format'

describe('formatBytes', () => {
  it('formats sub-kilobyte values as whole bytes', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(512)).toBe('512 B')
  })

  it('scales through binary units', () => {
    expect(formatBytes(1024)).toBe('1.0 KB')
    expect(formatBytes(1_887_437)).toBe('1.8 MB')
    expect(formatBytes(9_122_611)).toBe('8.7 MB')
  })

  it('returns a placeholder for invalid input instead of NaN', () => {
    expect(formatBytes(-1)).toBe('—')
    expect(formatBytes(Number.NaN)).toBe('—')
  })
})

describe('formatDuration', () => {
  it('uses milliseconds below one second', () => {
    expect(formatDuration(340)).toBe('340ms')
  })

  it('uses two decimal seconds below one minute', () => {
    expect(formatDuration(8420)).toBe('8.42s')
  })

  it('uses minutes and padded seconds above one minute', () => {
    expect(formatDuration(64_000)).toBe('1m 04s')
  })
})

describe('dimension helpers', () => {
  it('formats dimensions with a multiplication sign', () => {
    expect(formatDimensions(1280, 720)).toBe('1,280 × 720')
  })

  it('formats megapixels to one decimal', () => {
    expect(formatMegapixels(5120, 2880)).toBe('14.7 MP')
  })
})
