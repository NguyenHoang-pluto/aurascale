import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import { formatBytes, formatMegapixelCount } from '@/lib/format'
import type { ErrorCode, ProblemDetail } from '@/types/api'

/**
 * Turning a backend error into something a user can read, in their language.
 *
 * The backend writes its `detail` in English and will keep doing so: the API
 * contract is language-neutral and stable, which is exactly what makes this
 * possible. What travels is the `code` — one of a fixed set — plus a `context`
 * object of the numbers that matter. This module maps that pair onto a
 * translated sentence.
 *
 * The English `detail` is not thrown away. For a code this build does not know
 * about, it is shown as supplementary technical information, because a
 * sentence in the wrong language is more useful than no sentence at all.
 */

/** Every code the backend and the API client can produce. */
export const ERROR_CODES: readonly ErrorCode[] = [
  'unsupported_format',
  'corrupted_image',
  'file_too_large',
  'image_too_large',
  'image_too_small',
  'output_too_large',
  'invalid_parameters',
  'out_of_memory',
  'gpu_unavailable',
  'storage_full',
  'model_not_found',
  'model_load_failed',
  'model_download_failed',
  'inference_failed',
  'job_not_found',
  'job_not_completed',
  'job_cancelled',
  'queue_full',
  'timeout',
  'internal_error',
  'network_error',
  'backend_unavailable',
  'request_cancelled',
  'malformed_response',
]

/**
 * The values each message needs before it reads as a sentence.
 *
 * The backend normally sends these in `context`, but it is not obliged to, and
 * an older build might not. Without this table a missing value would silently
 * become an empty string — "That image is , which is larger than the  limit."
 * — so each of these codes also has a `detailGeneric` wording that names no
 * numbers, and that one is used whenever a value is absent.
 */
const REQUIRED_VALUES: Partial<Record<ErrorCode, readonly string[]>> = {
  file_too_large: ['limit'],
  image_too_large: ['actual', 'limit'],
  image_too_small: ['minimum'],
  output_too_large: ['projected', 'limit'],
  model_not_found: ['requested'],
}

export interface TranslatedError {
  title: string
  detail: string
  /** Developer detail, deliberately left in the language the server wrote it. */
  technical: string | undefined
  code: string
}

function isKnownCode(code: string): code is ErrorCode {
  return (ERROR_CODES as readonly string[]).includes(code)
}

/**
 * Interpolation values for a code, drawn from the backend's own context.
 *
 * Raw numbers are turned into the units the message talks about — bytes into
 * "32 MB", pixel counts into "16 MP" — because a translated sentence should
 * not contain `16000000`.
 */
export function interpolationFor(
  problem: Pick<ProblemDetail, 'code' | 'context'>,
  locale?: string,
): Record<string, string | number> {
  const context = problem.context ?? {}
  const values: Record<string, string | number> = {}

  const number = (key: string): number | undefined => {
    const value = context[key]
    return typeof value === 'number' ? value : undefined
  }

  const bytes = number('limitBytes')
  if (bytes !== undefined) values['limit'] = formatBytes(bytes, 0, locale)

  const limitPixels = number('limitPixels')
  if (limitPixels !== undefined) values['limit'] = formatMegapixelCount(limitPixels, locale)

  const actualPixels = number('actualPixels')
  if (actualPixels !== undefined) values['actual'] = formatMegapixelCount(actualPixels, locale)

  const projected = number('projectedPixels')
  if (projected !== undefined) values['projected'] = formatMegapixelCount(projected, locale)

  const minimum = number('minimumDimension')
  if (minimum !== undefined) values['minimum'] = minimum

  const requested = context['requested']
  if (typeof requested === 'string') values['requested'] = requested

  const model = context['model']
  if (typeof model === 'string') values['model'] = model

  const status = context['status']
  if (typeof status === 'string') values['status'] = status

  return values
}

/** Translate one problem document. Pure, so it can be tested without a render. */
export function translateProblem(
  t: TFunction,
  problem: Pick<ProblemDetail, 'code' | 'detail' | 'context'> & { technical?: string | null },
  locale?: string,
): TranslatedError {
  const known = isKnownCode(problem.code)
  const key = known ? problem.code : 'unknown'
  const values = interpolationFor(problem, locale)

  const required = known ? (REQUIRED_VALUES[problem.code] ?? []) : []
  const complete = required.every((name) => name in values)
  const detailKey = complete ? `${key}.detail` : `${key}.detailGeneric`

  return {
    title: t(`${key}.title`, { ns: 'errors' }),
    detail: t(detailKey, { ns: 'errors', ...values }),
    // An unknown code keeps the server's own sentence as the technical note,
    // so nothing the backend said is lost.
    technical: known
      ? (problem.technical ?? undefined)
      : (problem.technical ?? problem.detail),
    code: problem.code,
  }
}

/** Hook form, for components. */
export function useErrorMessage(): (
  problem: Pick<ProblemDetail, 'code' | 'detail' | 'context'> & { technical?: string | null },
) => TranslatedError {
  const { t, i18n } = useTranslation('errors')

  return (problem) => translateProblem(t, problem, i18n.language)
}
