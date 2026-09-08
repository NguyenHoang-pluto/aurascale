import i18next from 'i18next'
import { initReactI18next } from 'react-i18next'
import {
  DEFAULT_LANGUAGE,
  isLanguage,
  resolveInitialLanguage,
  useLanguageStore,
  type Language,
} from '@/stores/useLanguageStore'
import { DEFAULT_NAMESPACE, NAMESPACES, resources } from './resources'

/**
 * Translation setup.
 *
 * There is no language-detector plugin on purpose. `useLanguageStore` already
 * persists the choice through the same zustand middleware the theme uses, and
 * a second mechanism reading a second storage key could disagree with it about
 * what the user last picked.
 */

export function createI18n() {
  void i18next.use(initReactI18next).init({
    resources,
    lng: DEFAULT_LANGUAGE,
    fallbackLng: DEFAULT_LANGUAGE,
    ns: [...NAMESPACES],
    defaultNS: DEFAULT_NAMESPACE,
    interpolation: {
      // React escapes everything it renders, so escaping here would double-
      // encode any apostrophe or ampersand in a translation.
      escapeValue: false,
    },
    react: {
      // Resources are bundled and present before the first render, so there is
      // nothing to suspend on; enabling it would add a loading state to every
      // component for data that never arrives late.
      useSuspense: false,
    },
    // An empty string in a locale file is a mistake, not a deliberate blank;
    // falling through to English is more useful than rendering nothing.
    returnEmptyString: false,
  })

  // `document.lang` follows i18next rather than only the switcher, so any
  // path that changes the language -- including a test resetting it -- leaves
  // the document describing itself correctly.
  i18next.on('languageChanged', (language) => {
    if (isLanguage(language)) applyDocumentLanguage(language)
  })

  return i18next
}

/** Point the document at the active language, for screen readers and hyphenation. */
export function applyDocumentLanguage(language: Language): void {
  if (typeof document !== 'undefined') {
    document.documentElement.lang = language
  }
}

/**
 * Apply the stored (or detected) language before the first render.
 *
 * Called from `main.tsx` beside `initTheme()`, so the first painted frame is
 * already in the right language rather than flashing English.
 */
export function initLanguage(): void {
  const language = resolveInitialLanguage()

  useLanguageStore.setState({ language })
  void i18next.changeLanguage(language)
  applyDocumentLanguage(language)
}

/** Switch language everywhere: i18next, the persisted store, and the document. */
export function changeLanguage(language: Language): void {
  useLanguageStore.getState().setLanguage(language)
  void i18next.changeLanguage(language)
  applyDocumentLanguage(language)
}

/** The active language, for the formatters that need a locale. */
export function currentLanguage(): Language {
  return useLanguageStore.getState().language
}

export const i18n = createI18n()
