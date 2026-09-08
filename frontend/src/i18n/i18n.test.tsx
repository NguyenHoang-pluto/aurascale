import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { LanguageSwitcher } from '@/components/layout/LanguageSwitcher'
import { TopNav } from '@/components/layout/TopNav'
import { WorkspaceLayout } from '@/components/layout/WorkspaceLayout'
import { formatBytes, formatDimensions, formatNumber } from '@/lib/format'
import {
  DEFAULT_LANGUAGE,
  LANGUAGES,
  STORAGE_KEY,
  detectBrowserLanguage,
  readStoredLanguage,
  resolveInitialLanguage,
  useLanguageStore,
} from '@/stores/useLanguageStore'
import { renderWithProviders } from '@/test/renderWithProviders'
import type { ErrorCode } from '@/types/api'
import { changeLanguage, currentLanguage, i18n } from '.'
import { ERROR_CODES, interpolationFor, translateProblem } from './errorMessages'
import { modelDescription, modelName } from './modelMessages'
import { NAMESPACES, TECHNICAL_TERMS, resources } from './resources'

/**
 * The translation layer.
 *
 * These tests are about the mechanism, not the wording: that both languages
 * define the same keys, that a switch reaches the rendered output, that a
 * choice survives a remount, that every backend error code has a sentence, and
 * that the things which must not be translated were not.
 */

// --------------------------------------------------------------- key parity

/** Every leaf key in an object, as dotted paths. */
function leafKeys(value: unknown, prefix = ''): string[] {
  if (typeof value !== 'object' || value === null) return [prefix]

  return Object.entries(value).flatMap(([key, child]) =>
    leafKeys(child, prefix === '' ? key : `${prefix}.${key}`),
  )
}

describe('key parity', () => {
  it.each(NAMESPACES)('%s defines the same keys in both languages', (namespace) => {
    const en = leafKeys(resources.en[namespace]).sort()
    const vi = leafKeys(resources.vi[namespace]).sort()

    // Reported as set differences rather than a bare inequality, so a failure
    // names the key that is missing instead of dumping two long lists.
    expect(vi.filter((key) => !en.includes(key))).toEqual([])
    expect(en.filter((key) => !vi.includes(key))).toEqual([])
  })

  it('carries the same interpolation placeholders in both languages', () => {
    const placeholders = (text: string) =>
      [...text.matchAll(/{{(\w+)}}/g)].map((match) => match[1]).sort()

    for (const namespace of NAMESPACES) {
      for (const key of leafKeys(resources.en[namespace])) {
        const en = i18n.getResource('en', namespace, key) as unknown
        const vi = i18n.getResource('vi', namespace, key) as unknown
        if (typeof en !== 'string' || typeof vi !== 'string') continue

        // A translation that dropped a placeholder would render a sentence
        // with a hole in it, which no type or lint rule would catch.
        expect(placeholders(vi), `${namespace}:${key}`).toEqual(placeholders(en))
      }
    }
  })

  it('leaves no message empty', () => {
    for (const language of LANGUAGES) {
      for (const namespace of NAMESPACES) {
        for (const key of leafKeys(resources[language][namespace])) {
          const value = i18n.getResource(language, namespace, key) as unknown
          expect(String(value).trim(), `${language}/${namespace}:${key}`).not.toBe('')
        }
      }
    }
  })
})

// ------------------------------------------------------------ the mechanism

describe('language selection', () => {
  it('starts in English', () => {
    expect(currentLanguage()).toBe(DEFAULT_LANGUAGE)
    expect(i18n.language).toBe('en')
  })

  it('switches to Vietnamese and back', () => {
    changeLanguage('vi')
    expect(i18n.t('nav:links.history')).toBe('Lịch sử')

    changeLanguage('en')
    expect(i18n.t('nav:links.history')).toBe('History')
  })

  it('points the document at the active language', () => {
    changeLanguage('vi')
    expect(document.documentElement.lang).toBe('vi')

    changeLanguage('en')
    expect(document.documentElement.lang).toBe('en')
  })

  it('persists the choice where the next visit will find it', () => {
    changeLanguage('vi')

    expect(readStoredLanguage()).toBe('vi')
    expect(resolveInitialLanguage()).toBe('vi')
  })

  it('reads the browser preference only when nothing was stored', () => {
    // Nothing stored: the browser decides.
    expect(detectBrowserLanguage(['vi-VN', 'en-US'])).toBe('vi')
    expect(resolveInitialLanguage()).toBe('en')

    changeLanguage('vi')
    useLanguageStore.setState({ language: 'vi' })

    // Stored: the browser no longer gets a say.
    expect(resolveInitialLanguage()).toBe('vi')
  })

  it('ignores a language it does not have', () => {
    expect(detectBrowserLanguage(['fr-FR', 'de'])).toBeNull()
    expect(detectBrowserLanguage(undefined)).toBeNull()
  })

  it('survives a corrupt stored value rather than failing to start', () => {
    localStorage.setItem(STORAGE_KEY, '{ not json')

    expect(readStoredLanguage()).toBeNull()
    expect(resolveInitialLanguage()).toBe('en')
  })

  it('falls back to English for a key Vietnamese does not define', () => {
    // Added at runtime in a throwaway namespace rather than left in a locale
    // file: the parity test guarantees no English-only key exists, so this is
    // the only way to prove `fallbackLng` catches one that a later release
    // adds before its Vietnamese wording lands.
    i18n.addResourceBundle('en', '__probe__', { greeting: 'English only' })
    changeLanguage('vi')

    expect(i18n.t('__probe__:greeting')).toBe('English only')

    i18n.removeResourceBundle('en', '__probe__')
  })

  it('returns the key, not an empty string, for a message nothing defines', () => {
    // A blank would look like a rendering bug; the key at least says which
    // message is missing.
    expect(i18n.t('errors:__missing__.detail')).toBe('__missing__.detail')
  })
})

// ------------------------------------------------------------------- errors

describe('backend errors', () => {
  it.each(LANGUAGES)('gives every error code a title and a detail in %s', (language) => {
    for (const code of ERROR_CODES) {
      const title = i18n.getResource(language, 'errors', `${code}.title`) as unknown
      const detail = i18n.getResource(language, 'errors', `${code}.detail`) as unknown

      expect(typeof title, `${language} ${code}.title`).toBe('string')
      expect(typeof detail, `${language} ${code}.detail`).toBe('string')
    }
  })

  it('builds the sentence from the code and context, not the server text', () => {
    const message = translateProblem(
      i18n.t,
      {
        code: 'image_too_large',
        detail: 'That image is 81 MP, which is larger than the 16 MP limit.',
        context: { actualPixels: 81_000_000, limitPixels: 16_000_000 },
      },
      'en',
    )

    expect(message.title).toBe('Image too large')
    expect(message.detail).toBe('That image is 81.0 MP, which is larger than the 16.0 MP limit.')
    // Nothing of the server's own wording is promoted to the user.
    expect(message.technical).toBeUndefined()
  })

  it('translates the same code into Vietnamese', () => {
    changeLanguage('vi')

    const message = translateProblem(
      i18n.t,
      {
        code: 'image_too_large',
        detail: 'That image is 81 MP, which is larger than the 16 MP limit.',
        context: { actualPixels: 81_000_000, limitPixels: 16_000_000 },
      },
      'vi',
    )

    expect(message.detail).not.toContain('larger than')
    // Vietnamese groups digits with "." and marks decimals with ",".
    expect(message.detail).toContain('81,0 MP')
  })

  it('says something whole when the context is missing', () => {
    const message = translateProblem(
      i18n.t,
      { code: 'image_too_large', detail: 'That image is too big.' },
      'en',
    )

    // The interpolating wording would have rendered "That image is , which is
    // larger than the  limit."
    expect(message.detail).not.toContain('  ')
    expect(message.detail).toBe('That image has more pixels than one job will process.')
  })

  it('keeps the server sentence as technical detail for a code it does not know', () => {
    const message = translateProblem(
      i18n.t,
      // Cast because the type only admits codes this build knows; the point
      // of the test is a server that is newer than the type.
      { code: 'something_new' as ErrorCode, detail: 'A newer backend said this.' },
      'en',
    )

    expect(message.detail).toBe(i18n.t('errors:unknown.detail'))
    expect(message.technical).toBe('A newer backend said this.')
    expect(message.code).toBe('something_new')
  })

  it('converts raw context numbers into the units the sentence talks about', () => {
    const values = interpolationFor(
      {
        code: 'file_too_large',
        context: { limitBytes: 33_554_432 },
      },
      'en',
    )

    expect(values['limit']).toBe('32 MB')
  })
})

// -------------------------------------------------------------- formatting

describe('locale-aware formatting', () => {
  it('groups and marks decimals the way each language does', () => {
    expect(formatNumber(1280, 'en')).toBe('1,280')
    expect(formatNumber(1280, 'vi')).toBe('1.280')
    expect(formatNumber(8.42, 'en', 2)).toBe('8.42')
    expect(formatNumber(8.42, 'vi', 2)).toBe('8,42')
  })

  it('keeps unit symbols identical, because they are not words', () => {
    expect(formatBytes(33_554_432, 0, 'vi')).toContain('MB')
    expect(formatDimensions(1280, 960, 'vi')).toBe('1.280 × 960')
  })
})

// ------------------------------------------------------------- model registry

describe('model descriptions', () => {
  /** Exactly as `/api/models` serialises it, English description included. */
  const X4PLUS = {
    id: 'RealESRGAN_x4plus',
    name: 'Real-ESRGAN x4 Plus',
    description: 'General-purpose 4x upscaler. Best default for photographs.',
  }

  it('reads the English description in English', () => {
    expect(modelDescription(i18n.t, X4PLUS)).toBe(
      'General-purpose 4x upscaler. Best default for photographs.',
    )
  })

  it('reads the Vietnamese description in Vietnamese', () => {
    changeLanguage('vi')

    const detail = modelDescription(i18n.t, X4PLUS)
    expect(detail).toContain('đa dụng')
    expect(detail).not.toBe(X4PLUS.description)
  })

  it('describes every model the manifest publishes, in both languages', () => {
    const ids = Object.keys(resources.en.models)

    // The five entries in models/manifest.json. A model added to the registry
    // without wording here would silently fall back to English.
    expect(ids).toContain('RealESRGAN_x4plus')
    expect(ids).toContain('RealESRGAN_x4plus_anime_6B')
    expect(ids).toContain('RealESRGAN_x2plus')
    expect(ids).toContain('realesr-general-x4v3')
    expect(ids).toContain('realesr-general-wdn-x4v3')

    for (const id of ids) {
      for (const language of LANGUAGES) {
        const value = i18n.getResource(language, 'models', `${id}.description`) as unknown
        expect(typeof value, `${language} ${id}`).toBe('string')
        expect(String(value).trim(), `${language} ${id}`).not.toBe('')
      }
    }
  })

  it('gives a different sentence in each language for every model', () => {
    for (const id of Object.keys(resources.en.models)) {
      const en = i18n.getResource('en', 'models', `${id}.description`) as string
      const vi = i18n.getResource('vi', 'models', `${id}.description`) as string

      // A Vietnamese entry left as a copy of the English one is an untranslated
      // string that no other check would notice.
      expect(vi, id).not.toBe(en)
    }
  })

  it('never translates the model name', () => {
    changeLanguage('vi')
    expect(modelName(X4PLUS)).toBe('Real-ESRGAN x4 Plus')

    changeLanguage('en')
    expect(modelName(X4PLUS)).toBe('Real-ESRGAN x4 Plus')
  })

  it('keeps the technical identifiers inside the Vietnamese wording', () => {
    // The DNI pair is named by id in both languages; translating "wdn" or
    // "realesr-general-x4v3" would stop it matching the manifest.
    const vi = i18n.getResource('vi', 'models', 'realesr-general-x4v3.description') as string

    expect(vi).toContain('DNI')
    expect(vi).toContain('wdn')
  })

  it('falls back to the server sentence for a model this build has no wording for', () => {
    changeLanguage('vi')

    const unknown = {
      id: 'RealESRGAN_x8plus_future',
      name: 'Real-ESRGAN x8 Plus',
      description: 'A model added after this build shipped.',
    }

    // Not machine-translated, not blank, and not a raw key.
    expect(modelDescription(i18n.t, unknown)).toBe('A model added after this build shipped.')
  })
})

// -------------------------------------------------------- technical terms

describe('technical terms', () => {
  it('never translates an identifier or a unit symbol', () => {
    const enText = JSON.stringify(resources.en)
    const viText = JSON.stringify(resources.vi)

    for (const term of TECHNICAL_TERMS) {
      // A term the English copy uses must appear verbatim in the Vietnamese
      // copy too: translating "CUDA" or "PNG" would stop it matching what the
      // backend, the manifest and the docs call it.
      if (enText.includes(term)) {
        expect(viText, term).toContain(term)
      }
    }
  })

  it('keeps the product name as it is', () => {
    changeLanguage('vi')
    expect(i18n.t('nav:brand')).toContain('PixelForge AI')
  })
})

// ------------------------------------------------------------ the switcher

describe('LanguageSwitcher', () => {
  it('is reachable by its accessible name', async () => {
    renderWithProviders(<LanguageSwitcher />)

    expect(await screen.findByRole('combobox', { name: 'Language' })).toBeInTheDocument()
  })

  it('names each language in that language, so it can be found', async () => {
    const user = userEvent.setup()
    renderWithProviders(<LanguageSwitcher />)

    await user.click(screen.getByRole('combobox', { name: 'Language' }))

    const list = within(await screen.findByRole('listbox'))
    expect(list.getByRole('option', { name: 'English' })).toBeInTheDocument()
    expect(list.getByRole('option', { name: 'Tiếng Việt' })).toBeInTheDocument()
  })

  it('switches the interface it is rendered in', async () => {
    const user = userEvent.setup()
    renderWithProviders(
      <>
        <LanguageSwitcher />
        <TopNav />
      </>,
    )

    expect(screen.getByRole('link', { name: 'History' })).toBeInTheDocument()

    await user.click(screen.getAllByRole('combobox', { name: 'Language' })[0]!)
    await user.click(await screen.findByRole('option', { name: 'Tiếng Việt' }))

    await waitFor(() => {
      expect(screen.getByRole('link', { name: 'Lịch sử' })).toBeInTheDocument()
    })
    expect(screen.queryByRole('link', { name: 'History' })).not.toBeInTheDocument()
    expect(currentLanguage()).toBe('vi')
  })

  it('remembers the choice across a remount', async () => {
    const user = userEvent.setup()
    const first = renderWithProviders(<LanguageSwitcher />)

    await user.click(screen.getByRole('combobox', { name: 'Language' }))
    await user.click(await screen.findByRole('option', { name: 'Tiếng Việt' }))
    await waitFor(() => { expect(currentLanguage()).toBe('vi') })
    first.unmount()

    renderWithProviders(<TopNav />)

    expect(await screen.findByRole('link', { name: 'Lịch sử' })).toBeInTheDocument()
  })
})

// -------------------------------------------------- translated aria labels

describe('accessible names', () => {
  it('translates the names screen readers announce, not only visible text', async () => {
    changeLanguage('vi')
    renderWithProviders(<TopNav />)

    // The nav landmark and the theme button are announced but never read on
    // screen, so an untranslated one would be invisible to a sighted reviewer.
    expect(await screen.findByRole('navigation', { name: 'Chính' })).toBeInTheDocument()
    // Either direction, depending on the theme the stub resolves to; what
    // matters is that the label is Vietnamese rather than English.
    expect(
      screen.getByRole('button', { name: /Chuyển sang giao diện/ }),
    ).toBeInTheDocument()
  })

  it('translates the workspace landmarks', async () => {
    changeLanguage('vi')
    renderWithProviders(<WorkspaceLayout canvas={<p>canvas</p>} rail={<p>rail</p>} />)

    expect(
      await screen.findByRole('region', { name: 'Khu làm việc với ảnh' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('complementary', { name: 'Thiết lập nâng cấp' }),
    ).toBeInTheDocument()
  })
})
