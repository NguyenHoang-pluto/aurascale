import enCommon from './locales/en/common.json'
import enCompare from './locales/en/compare.json'
import enEnhance from './locales/en/enhance.json'
import enErrors from './locales/en/errors.json'
import enHistory from './locales/en/history.json'
import enJob from './locales/en/job.json'
import enModels from './locales/en/models.json'
import enNav from './locales/en/nav.json'
import enSystem from './locales/en/system.json'
import enUpload from './locales/en/upload.json'
import enViewer from './locales/en/viewer.json'
import viCommon from './locales/vi/common.json'
import viCompare from './locales/vi/compare.json'
import viEnhance from './locales/vi/enhance.json'
import viErrors from './locales/vi/errors.json'
import viHistory from './locales/vi/history.json'
import viJob from './locales/vi/job.json'
import viModels from './locales/vi/models.json'
import viNav from './locales/vi/nav.json'
import viSystem from './locales/vi/system.json'
import viUpload from './locales/vi/upload.json'
import viViewer from './locales/vi/viewer.json'

/**
 * Every translation, bundled.
 *
 * Imported statically rather than fetched at runtime: both languages together
 * are a few kilobytes, and loading them over the network would mean a
 * suspense boundary, a loading state, and a flash of untranslated text on
 * every screen — for less than the size of one icon.
 */

/** Namespaces, one per feature area. Order is only for readability. */
export const NAMESPACES = [
  'common',
  'nav',
  'upload',
  'viewer',
  'enhance',
  'job',
  'compare',
  'history',
  'system',
  'models',
  'errors',
] as const

export type Namespace = (typeof NAMESPACES)[number]

/** Namespace used when a caller does not name one. */
export const DEFAULT_NAMESPACE: Namespace = 'common'

export const resources = {
  en: {
    common: enCommon,
    nav: enNav,
    upload: enUpload,
    viewer: enViewer,
    enhance: enEnhance,
    job: enJob,
    compare: enCompare,
    history: enHistory,
    system: enSystem,
    models: enModels,
    errors: enErrors,
  },
  vi: {
    common: viCommon,
    nav: viNav,
    upload: viUpload,
    viewer: viViewer,
    enhance: viEnhance,
    job: viJob,
    compare: viCompare,
    history: viHistory,
    system: viSystem,
    models: viModels,
    errors: viErrors,
  },
} as const

/**
 * Technical terms that are the same in every language.
 *
 * Documented here rather than left to each translator's judgement: a model id
 * or a unit symbol that got translated would stop matching what the backend
 * says, what the manifest says, and what the user reads in the docs.
 */
export const TECHNICAL_TERMS = [
  'PixelForge AI',
  'Real-ESRGAN',
  'RRDBNet',
  'SRVGGNetCompact',
  'CUDA',
  'PyTorch',
  'Python',
  'DNI',
  'VRAM',
  'GPU',
  'CPU',
  'SSE',
  'EXIF',
  'ICC',
  'fp16',
  'PNG',
  'JPEG',
  'WEBP',
  'MP',
  'B',
  'KB',
  'MB',
  'GB',
] as const
