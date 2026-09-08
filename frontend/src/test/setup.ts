import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach } from 'vitest'
import { DEFAULT_LANGUAGE, STORAGE_KEY, useLanguageStore } from '@/stores/useLanguageStore'
import { i18n } from '@/i18n'
import {
  installObjectUrl,
  installPointerApis,
  installResizeObserver,
  resetObjectUrl,
  resetResizeObserver,
} from './browserStubs'
import { installMatchMedia, resetMatchMedia } from './matchMedia'

beforeEach(() => {
  resetMatchMedia()
  installMatchMedia()
  resetObjectUrl()
  installObjectUrl()
  resetResizeObserver()
  installResizeObserver()
  installPointerApis()

  // i18next and the language store are module-level singletons, so a test that
  // switches to Vietnamese would otherwise leave every later test running in
  // it. Reset both, and the persisted key they read on startup, so each test
  // begins in English exactly as a first-time visitor would.
  localStorage.removeItem(STORAGE_KEY)
  useLanguageStore.setState({ language: DEFAULT_LANGUAGE })
  if (i18n.language !== DEFAULT_LANGUAGE) void i18n.changeLanguage(DEFAULT_LANGUAGE)
})

afterEach(() => {
  cleanup()
})
