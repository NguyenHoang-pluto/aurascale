/**
 * The Enhance rail in both languages, rendered rather than inspected.
 *
 * `i18n.test.tsx` already proves the translation layer: keys exist in both
 * languages, `modelDescription` maps an id onto a sentence, a model name is
 * never translated. What it cannot prove is that the components actually call
 * any of it, or that a language change reaches the screen without a reload.
 *
 * That is the gap this file covers. Everything here renders the real panels
 * against a stubbed `GET /api/models` and reads what a user would see.
 *
 * The two failure modes it is aimed at:
 *
 *   * a component that renders the backend's English `description` directly,
 *     which passes every key-parity check and still shows English to a
 *     Vietnamese reader;
 *   * a translated string captured once - in a `useMemo`, a module constant,
 *     or a `useState` initialiser - so the first language rendered is the only
 *     one the user ever gets.
 */

import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { EnhancePanels } from './EnhancePanels'
import { changeLanguage, i18n } from '@/i18n'
import { renderWithProviders } from '@/test/renderWithProviders'
import { GPU_SYSTEM, HEALTH, stubApi } from '@/test/systemFixtures'
import { DEFAULT_DENOISE, DEFAULT_QUALITY, DEFAULT_SCALE, useEnhancementStore } from '@/stores/useEnhancementStore'
import { useWorkspaceStore } from '@/stores/useWorkspaceStore'
import type { ModelInfo } from '@/types/system'

/**
 * Two models with English descriptions, exactly as the backend sends them.
 *
 * The English text here is deliberately the manifest's own wording. If a panel
 * renders `description` straight through, the English assertions still pass and
 * only the Vietnamese ones fail - which is the point.
 */
const DENOISE_MODEL: ModelInfo = {
  id: 'realesr-general-x4v3',
  name: 'realesr-general-x4v3',
  description:
    'Compact 4x model. Paired with the wdn variant it provides a real denoise-strength control via DNI weight interpolation.',
  arch: 'SRVGGNet',
  scale: 4,
  supportsDenoise: true,
  supportedScales: [4, 8, 16],
  downloaded: true,
  sizeMb: 4.7,
}

const PLAIN_MODEL: ModelInfo = {
  id: 'RealESRGAN_x4plus',
  name: 'Real-ESRGAN x4 Plus',
  description: 'General-purpose 4x upscaler. Best default for photographs.',
  arch: 'RRDBNet',
  scale: 4,
  supportsDenoise: false,
  supportedScales: [4, 8, 16],
  downloaded: true,
  sizeMb: 63.9,
}

function stubModels(models: ModelInfo[] = [DENOISE_MODEL, PLAIN_MODEL]) {
  return stubApi({ '/api/health': HEALTH, '/api/system': GPU_SYSTEM, '/api/models': models })
}

function loadImage() {
  useWorkspaceStore.setState({
    source: {
      file: new File([new Uint8Array(1000)], 'photo.png', { type: 'image/png' }),
      objectUrl: 'blob:test',
      metadata: { name: 'photo.png', width: 1280, height: 720, sizeBytes: 1000, format: 'PNG' },
    },
    problem: null,
    isLoading: false,
  })
}

/** The description a locale file defines for a model, by its backend id. */
function wording(language: 'en' | 'vi', id: string): string {
  return i18n.getResource(language, 'models', `${id}.description`) as string
}

/** A namespaced Enhance string, in one language, without switching to it. */
function enhanceText(language: 'en' | 'vi', key: string): string {
  return i18n.getResource(language, 'enhance', key) as string
}

/**
 * Open the model dropdown and return its listbox.
 *
 * The per-model wording lives on the options, so it is only in the document
 * while the menu is open. Reading it any other way would test the translation
 * layer again rather than what the rail renders.
 */
async function openModelMenu(): Promise<HTMLElement> {
  const trigger = await screen.findByRole('combobox', {
    name: new RegExp(enhanceText(i18n.language === 'vi' ? 'vi' : 'en', 'model.label'), 'i'),
  })
  await userEvent.click(trigger)
  return screen.findByRole('listbox')
}

beforeEach(async () => {
  await changeLanguage('en')
  useWorkspaceStore.setState({ source: null, problem: null, isLoading: false })
  useEnhancementStore.setState({
    modelId: null,
    modelChosen: false,
    scale: DEFAULT_SCALE,
    denoiseStrength: DEFAULT_DENOISE,
    denoiseChosen: false,
    sharpenStrength: 0,
    format: 'png',
    quality: DEFAULT_QUALITY,
    preserveMetadata: true,
    activeJobId: null,
  })
})

afterEach(async () => {
  vi.unstubAllGlobals()
  await changeLanguage('en')
})

// ------------------------------------------------------- model descriptions

describe('model descriptions in the rail', () => {
  it('shows the English wording from the locale file, not the backend field', async () => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const listbox = await openModelMenu()
    expect(within(listbox).getByText(wording('en', DENOISE_MODEL.id))).toBeInTheDocument()
  })

  it('shows the Vietnamese wording once the language changes', async () => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const before = await openModelMenu()
    expect(within(before).getByText(wording('en', DENOISE_MODEL.id))).toBeInTheDocument()

    await changeLanguage('vi')

    await waitFor(() => {
      expect(within(before).getByText(wording('vi', DENOISE_MODEL.id))).toBeInTheDocument()
    })
  })

  it('replaces the English sentence rather than showing both', async () => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const english = wording('en', DENOISE_MODEL.id)
    const listbox = await openModelMenu()
    expect(within(listbox).getByText(english)).toBeInTheDocument()

    await changeLanguage('vi')

    await waitFor(() => {
      expect(within(listbox).queryByText(english)).not.toBeInTheDocument()
    })
  })

  it('switches back to English without a reload', async () => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const listbox = await openModelMenu()
    expect(within(listbox).getByText(wording('en', DENOISE_MODEL.id))).toBeInTheDocument()

    await changeLanguage('vi')
    await waitFor(() => {
      expect(within(listbox).getByText(wording('vi', DENOISE_MODEL.id))).toBeInTheDocument()
    })

    await changeLanguage('en')
    await waitFor(() => {
      expect(within(listbox).getByText(wording('en', DENOISE_MODEL.id))).toBeInTheDocument()
    })
  })

  it('never translates the model name', async () => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    // The name matches the manifest, the weights file and the docs. A
    // translated one would stop matching all three.
    const listbox = await openModelMenu()
    expect(within(listbox).getByText(PLAIN_MODEL.name)).toBeInTheDocument()

    await changeLanguage('vi')

    await waitFor(() => {
      expect(within(listbox).getByText(PLAIN_MODEL.name)).toBeInTheDocument()
    })
  })
})

// ------------------------------------------------------------ the help texts

/**
 * The explanatory lines beside each control, which are the ones most likely to
 * be left in English: they are prose rather than labels, so a missing
 * translation reads as a design choice rather than a bug.
 */
const HELP_KEYS = [
  'model.description',
  'scale.descriptionTwoPass',
  'sharpen.description',
  'format.description',
  'quality.descriptionLossless',
  'mode.standardHint',
  'sizing.descriptionScale',
] as const

describe('control help text', () => {
  it.each(HELP_KEYS)('renders %s in English', async (key) => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    expect(await screen.findByText(enhanceText('en', key))).toBeInTheDocument()
  })

  it.each(HELP_KEYS)('renders %s in Vietnamese after a language change', async (key) => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    await screen.findByText(enhanceText('en', key))

    await changeLanguage('vi')

    expect(await screen.findByText(enhanceText('vi', key))).toBeInTheDocument()
  })

  it('localises the denoise explanation, which only one model shows', async () => {
    stubModels([DENOISE_MODEL])
    renderWithProviders(<EnhancePanels />)
    loadImage()

    await screen.findByText(enhanceText('en', 'denoise.description'))

    await changeLanguage('vi')

    expect(await screen.findByText(enhanceText('vi', 'denoise.description'))).toBeInTheDocument()
  })
})

// --------------------------------------------------------- unknown models

describe('a model this build has no wording for', () => {
  const FUTURE: ModelInfo = {
    id: 'realesr-future-x8v9',
    name: 'Real-ESRGAN Future x8',
    description: 'A model added after this build shipped.',
    arch: 'RRDBNet',
    scale: 4,
    supportsDenoise: false,
    supportedScales: [4],
    downloaded: true,
    sizeMb: 12.0,
  }

  it('falls back to the server sentence instead of showing a raw key', async () => {
    stubModels([FUTURE])
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const listbox = await openModelMenu()
    expect(within(listbox).getByText(FUTURE.description)).toBeInTheDocument()
    // The failure this guards: i18next echoing `models:<id>.description` back.
    expect(within(listbox).queryByText(/^models:/)).not.toBeInTheDocument()
  })

  it('still renders the selector, so one unknown model breaks nothing', async () => {
    stubModels([FUTURE, PLAIN_MODEL])
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const listbox = await openModelMenu()
    expect(within(listbox).getByText(FUTURE.name)).toBeInTheDocument()
    expect(within(listbox).getByText(wording('en', PLAIN_MODEL.id))).toBeInTheDocument()
  })

  it('keeps the fallback in place after a language change', async () => {
    stubModels([FUTURE])
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const listbox = await openModelMenu()
    expect(within(listbox).getByText(FUTURE.description)).toBeInTheDocument()

    await changeLanguage('vi')

    // Still the English sentence, because there is nothing else to show. A
    // sentence in the wrong language beats a blank line or a key.
    await waitFor(() => {
      expect(within(listbox).getByText(FUTURE.description)).toBeInTheDocument()
    })
  })
})

// ------------------------------------------------------------ the selector

describe('the model dropdown', () => {
  it('localises every option it lists', async () => {
    stubModels()
    renderWithProviders(<EnhancePanels />)
    loadImage()

    const listbox = await openModelMenu()
    for (const model of [DENOISE_MODEL, PLAIN_MODEL]) {
      expect(within(listbox).getByText(wording('en', model.id))).toBeInTheDocument()
      expect(within(listbox).getByText(model.name)).toBeInTheDocument()
    }
  })
})
