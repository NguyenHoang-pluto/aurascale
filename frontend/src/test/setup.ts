import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach } from 'vitest'
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
})

afterEach(() => {
  cleanup()
})
