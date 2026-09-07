/**
 * Upload limits enforced in the browser.
 *
 * These mirror the backend defaults in `app/core/config.py`. Client-side checks
 * exist to give immediate, specific feedback — they are a convenience, never a
 * security control; the backend re-validates everything it receives.
 *
 * `GET /api/system` reports hardware capability, not upload limits, so these
 * stay the client's own copy; the API contract in docs/api.md is what keeps
 * them in step with the backend.
 */
export interface UploadLimits {
  maxFileSizeBytes: number
  maxInputPixels: number
  minDimension: number
  formats: readonly string[]
}

export const DEFAULT_UPLOAD_LIMITS: UploadLimits = {
  maxFileSizeBytes: 32 * 1024 * 1024,
  maxInputPixels: 16_000_000,
  minDimension: 32,
  formats: ['JPEG', 'PNG', 'WEBP'],
}

/** `accept` attribute for the file input. Advisory only — content is sniffed. */
export const ACCEPTED_MIME_TYPES = 'image/jpeg,image/png,image/webp'

/** Extensions shown to the user, matching the formats above. */
export const ACCEPTED_EXTENSIONS = ['JPG', 'JPEG', 'PNG', 'WEBP'] as const
