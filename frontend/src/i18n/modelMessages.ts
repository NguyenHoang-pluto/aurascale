import { useTranslation } from 'react-i18next'
import type { TFunction } from 'i18next'
import type { ModelInfo } from '@/types/system'

/**
 * Turning a model from the registry into something a user can read.
 *
 * The same arrangement as `errorMessages`, for the same reason: the backend's
 * `description` is written in English and will keep being, because the model
 * manifest is one source of truth shared by the API, the downloader and the
 * docs. What travels is the stable `id`, and this module maps that onto a
 * translated sentence.
 *
 * Nothing is machine-translated at runtime. A model this build has no wording
 * for falls back to the English the server sent, which is more useful than a
 * blank line or a raw key.
 */

/**
 * The model's own name, never translated.
 *
 * "Real-ESRGAN x4 Plus" is a product name and an identifier: it matches the
 * manifest, the weights file and the docs, and a translated version would stop
 * matching all three.
 */
export function modelName(model: Pick<ModelInfo, 'name'>): string {
  return model.name
}

/**
 * The model's description in the active language.
 *
 * `id` is the key rather than a camelCase alias, following the `errors`
 * namespace, which is keyed by the backend's own error codes. An alias table
 * would be a second list of model identifiers to keep in step with the
 * manifest, and the one that drifted would fail silently.
 */
export function modelDescription(
  t: TFunction,
  model: Pick<ModelInfo, 'id' | 'description'>,
): string {
  const key = `models:${model.id}.description`
  const translated = t(key)

  // i18next echoes the key back when nothing defines it. That is the signal
  // that this build predates the model, so the server's own sentence stands in.
  return translated === key || translated === `${model.id}.description`
    ? model.description
    : translated
}

/** Hook form, for components. */
export function useModelDescription(): (
  model: Pick<ModelInfo, 'id' | 'description'>,
) => string {
  const { t } = useTranslation('models')

  return (model) => modelDescription(t, model)
}
