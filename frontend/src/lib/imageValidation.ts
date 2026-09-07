import { DEFAULT_UPLOAD_LIMITS, type UploadLimits } from '@/config/limits'
import type { ErrorCode, ProblemDetail } from '@/types/api'
import type { ImageMetadata, SupportedFormat } from '@/types/image'
import { formatBytes, formatDimensions, formatMegapixels } from './format'

/**
 * Client-side image validation.
 *
 * Mirrors the backend's ordering (`docs/image-processing.md` § 2): cheapest
 * checks first, and each check protects the next from hostile input. The file
 * extension and the browser-reported MIME type are never trusted — the format
 * is determined by inspecting the leading bytes, and the image is then actually
 * decoded, which is the only way to detect a truncated or corrupt file.
 */

export type ValidationResult =
  | { ok: true; metadata: ImageMetadata }
  | { ok: false; problem: ProblemDetail }

/** Decodes a file far enough to report its pixel dimensions. */
export type ImageDecoder = (file: File) => Promise<{ width: number; height: number }>

function problem(
  code: ErrorCode,
  title: string,
  detail: string,
  technical?: string,
): ProblemDetail {
  return {
    type: `https://pixelforge.ai/errors/${code}`,
    title,
    status: 0,
    code,
    detail,
    ...(technical !== undefined ? { technical } : {}),
  }
}

// -------------------------------------------------------------- magic bytes

const SIGNATURE_BYTES = 12

function startsWith(bytes: Uint8Array, signature: readonly number[], offset = 0): boolean {
  return signature.every((byte, index) => bytes[offset + index] === byte)
}

/**
 * Identify the format from the file header.
 *
 * Returns `null` for anything not recognised — a renamed `.exe`, a PDF, an
 * unsupported image format such as TIFF or AVIF.
 */
export function sniffFormat(bytes: Uint8Array): SupportedFormat | null {
  // JPEG: SOI marker followed by any marker byte.
  if (startsWith(bytes, [0xff, 0xd8, 0xff])) return 'JPEG'

  // PNG: 8-byte signature including the CRLF/EOF transfer check.
  if (startsWith(bytes, [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])) return 'PNG'

  // WEBP: RIFF container whose form type is WEBP, at offset 8.
  if (startsWith(bytes, [0x52, 0x49, 0x46, 0x46]) && startsWith(bytes, [0x57, 0x45, 0x42, 0x50], 8)) {
    return 'WEBP'
  }

  return null
}

// ----------------------------------------------------------------- decoding

/**
 * Default decoder.
 *
 * `createImageBitmap` performs a real decode, so a file with a valid header but
 * corrupt image data is rejected here rather than rendering as a broken image.
 * The bitmap is closed immediately; keeping it alive would pin the decoded
 * pixels, which for a 16 MP image is around 64 MB.
 */
export const decodeImage: ImageDecoder = async (file) => {
  if (typeof createImageBitmap === 'function') {
    const bitmap = await createImageBitmap(file)
    try {
      return { width: bitmap.width, height: bitmap.height }
    } finally {
      bitmap.close()
    }
  }

  // Fallback for environments without createImageBitmap.
  const objectUrl = URL.createObjectURL(file)
  try {
    return await new Promise<{ width: number; height: number }>((resolve, reject) => {
      const image = new Image()
      image.onload = () => { resolve({ width: image.naturalWidth, height: image.naturalHeight }) }
      image.onerror = () => { reject(new Error('The image could not be decoded.')) }
      image.src = objectUrl
    })
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

// --------------------------------------------------------------- validation

export interface ValidateOptions {
  limits?: UploadLimits
  /** Injectable for testing; defaults to a real decode. */
  decode?: ImageDecoder
}

export async function validateImageFile(
  file: File,
  options: ValidateOptions = {},
): Promise<ValidationResult> {
  const limits = options.limits ?? DEFAULT_UPLOAD_LIMITS
  const decode = options.decode ?? decodeImage

  if (file.size === 0) {
    return {
      ok: false,
      problem: problem(
        'corrupted_image',
        'Empty file',
        'That file is empty. It may not have finished copying.',
        `name=${file.name} size=0`,
      ),
    }
  }

  if (file.size > limits.maxFileSizeBytes) {
    return {
      ok: false,
      problem: problem(
        'file_too_large',
        'File too large',
        `That file is ${formatBytes(file.size)}, larger than the ${formatBytes(
          limits.maxFileSizeBytes,
        )} limit.`,
        `size=${String(file.size)} limit=${String(limits.maxFileSizeBytes)}`,
      ),
    }
  }

  const header = new Uint8Array(await file.slice(0, SIGNATURE_BYTES).arrayBuffer())
  const format = sniffFormat(header)

  if (format === null) {
    return {
      ok: false,
      problem: problem(
        'unsupported_format',
        'Unsupported file type',
        `That file is not a ${limits.formats.join(', ')} image. Its contents do not match any supported format.`,
        // The declared type is recorded because it disagreeing with the
        // contents is exactly the case worth seeing while debugging.
        `declaredType=${file.type || 'none'} header=${[...header.slice(0, 8)]
          .map((b) => b.toString(16).padStart(2, '0'))
          .join(' ')}`,
      ),
    }
  }

  let dimensions: { width: number; height: number }
  try {
    dimensions = await decode(file)
  } catch (cause) {
    return {
      ok: false,
      problem: problem(
        'corrupted_image',
        'Image could not be read',
        'That file looks like an image but could not be decoded. It may be truncated or damaged.',
        cause instanceof Error ? cause.message : String(cause),
      ),
    }
  }

  const { width, height } = dimensions

  if (width < limits.minDimension || height < limits.minDimension) {
    return {
      ok: false,
      problem: problem(
        'image_too_small',
        'Image too small',
        `That image is ${formatDimensions(width, height)}. Images must be at least ${String(
          limits.minDimension,
        )} pixels on each side to have detail worth reconstructing.`,
        `input=${String(width)}x${String(height)} min=${String(limits.minDimension)}`,
      ),
    }
  }

  const pixels = width * height
  if (pixels > limits.maxInputPixels) {
    return {
      ok: false,
      problem: problem(
        'image_too_large',
        'Image too large',
        `That image is ${formatMegapixels(width, height)}, larger than the ${(
          limits.maxInputPixels / 1_000_000
        ).toFixed(0)} MP limit.`,
        `input=${String(width)}x${String(height)} pixels=${String(pixels)} limit=${String(
          limits.maxInputPixels,
        )}`,
      ),
    }
  }

  return {
    ok: true,
    metadata: { name: file.name, width, height, sizeBytes: file.size, format },
  }
}
