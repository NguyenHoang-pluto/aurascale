/**
 * Human-readable formatters shared across the workspace UI.
 *
 * Each takes an optional `locale`. Passing `undefined` keeps the previous
 * behaviour of following the browser, which is what the tests rely on; the UI
 * passes the language the user actually chose, because Vietnamese groups
 * digits with `.` where English uses `,` and a number formatted the wrong way
 * reads as a different number.
 *
 * Unit symbols are never translated: `MB`, `s`, `MP` and `×` are the same in
 * both languages, and inventing local equivalents would only obscure them.
 */

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const

/**
 * Format a byte count using binary units (1 KB = 1024 B).
 * Negative or non-finite input returns a neutral placeholder rather than NaN.
 */
export function formatBytes(bytes: number, fractionDigits = 1, locale?: string): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${formatNumber(Math.round(bytes), locale)} B`

  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < BYTE_UNITS.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${formatNumber(value, locale, fractionDigits)} ${BYTE_UNITS[unitIndex]}`
}

/** Format a millisecond duration as a compact human string (e.g. "8.42s", "1m 04s"). */
export function formatDuration(ms: number, locale?: string): string {
  if (!Number.isFinite(ms) || ms < 0) return '—'
  if (ms < 1000) return `${formatNumber(Math.round(ms), locale)}ms`

  const totalSeconds = ms / 1000
  if (totalSeconds < 60) return `${formatNumber(totalSeconds, locale, 2)}s`

  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${formatNumber(minutes, locale)}m ${String(seconds).padStart(2, '0')}s`
}

/** Format pixel dimensions the way the info panel shows them: "1280 × 720". */
export function formatDimensions(width: number, height: number, locale?: string): string {
  return `${formatNumber(width, locale)} × ${formatNumber(height, locale)}`
}

/** Format a megapixel count, used for the image-size guard messaging. */
export function formatMegapixels(width: number, height: number, locale?: string): string {
  return formatMegapixelCount(width * height, locale)
}

/** The same, from a pixel total — what the backend sends in an error context. */
export function formatMegapixelCount(pixels: number, locale?: string): string {
  return `${formatNumber(pixels / 1_000_000, locale, 1)} MP`
}

/**
 * One place where a number becomes text.
 *
 * `Intl.NumberFormat` rather than `toFixed` plus `toLocaleString`, so the
 * decimal separator and the grouping separator agree with each other. They do
 * not in Vietnamese if the two are applied separately.
 */
export function formatNumber(value: number, locale?: string, fractionDigits = 0): string {
  if (!Number.isFinite(value)) return '—'

  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(value)
}
