/** Image domain types shared by the upload and viewer features. */

export type SupportedFormat = 'JPEG' | 'PNG' | 'WEBP'

export interface ImageMetadata {
  /** Original file name, as supplied by the browser. */
  name: string
  width: number
  height: number
  sizeBytes: number
  /** Format determined by inspecting file contents, never the extension. */
  format: SupportedFormat
}

/**
 * An image the user has loaded into the workspace.
 *
 * `objectUrl` is owned by the store that created it and must be revoked when
 * the image is replaced or cleared, or the blob leaks for the page's lifetime.
 */
export interface LoadedImage {
  file: File
  objectUrl: string
  metadata: ImageMetadata
}
