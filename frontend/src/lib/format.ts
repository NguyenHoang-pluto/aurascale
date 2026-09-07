/** Human-readable formatters shared across the workspace UI. */

const BYTE_UNITS = ['B', 'KB', 'MB', 'GB', 'TB'] as const

/**
 * Format a byte count using binary units (1 KB = 1024 B).
 * Negative or non-finite input returns a neutral placeholder rather than NaN.
 */
export function formatBytes(bytes: number, fractionDigits = 1): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${Math.round(bytes)} B`

  let value = bytes
  let unitIndex = 0
  while (value >= 1024 && unitIndex < BYTE_UNITS.length - 1) {
    value /= 1024
    unitIndex += 1
  }
  return `${value.toFixed(fractionDigits)} ${BYTE_UNITS[unitIndex]}`
}

/** Format a millisecond duration as a compact human string (e.g. "8.42s", "1m 04s"). */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`

  const totalSeconds = ms / 1000
  if (totalSeconds < 60) return `${totalSeconds.toFixed(2)}s`

  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}m ${String(seconds).padStart(2, '0')}s`
}

/** Format pixel dimensions the way the info panel shows them: "1280 × 720". */
export function formatDimensions(width: number, height: number): string {
  return `${width.toLocaleString()} × ${height.toLocaleString()}`
}

/** Format a megapixel count, used for the image-size guard messaging. */
export function formatMegapixels(width: number, height: number): string {
  return `${((width * height) / 1_000_000).toFixed(1)} MP`
}
