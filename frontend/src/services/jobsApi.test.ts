import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { CreateJobRequest } from '@/types/job'

const apiRequest = vi.fn()

vi.mock('./apiClient', () => ({
  apiRequest: (...args: unknown[]) => apiRequest(...args),
  toApiError: (error: unknown) => error,
}))

const { createJob } = await import('./jobsApi')

/**
 * What actually goes on the wire.
 *
 * Phase 4 F1 found Enhancement Mode producing byte-identical output to
 * Standard, because the client always sent an explicit `model` and so the
 * backend's mode defaults could never apply - `resolve_model` is documented to
 * let an explicit choice win, and the client never stopped supplying one.
 *
 * These assert the field is present exactly when the user chose it, which is
 * the whole of the fix on this side.
 */

function submitted(): FormData {
  expect(apiRequest).toHaveBeenCalledTimes(1)
  const [, options] = apiRequest.mock.calls[0] as [string, { body: FormData }]
  return options.body
}

function request(overrides: Partial<CreateJobRequest> = {}): CreateJobRequest {
  return {
    file: new File([new Uint8Array([1, 2, 3])], 'in.png', { type: 'image/png' }),
    scale: 4,
    format: 'png',
    preserveMetadata: true,
    settings: {},
    ...overrides,
  }
}

beforeEach(() => {
  apiRequest.mockReset()
  apiRequest.mockResolvedValue({ jobId: 'j1', status: 'queued' })
})

describe('createJob model field', () => {
  it('omits model when the user has not chosen one, so the mode can decide', async () => {
    await createJob(request({ mode: 'creative' }))

    expect(submitted().has('model')).toBe(false)
    expect(submitted().get('mode')).toBe('creative')
  })

  it('sends model when the user did choose one', async () => {
    await createJob(request({ model: 'RealESRGAN_x2plus', mode: 'creative' }))

    expect(submitted().get('model')).toBe('RealESRGAN_x2plus')
  })

  it('still sends every field that is not conditional', async () => {
    await createJob(request())
    const form = submitted()

    expect(form.get('format')).toBe('png')
    expect(form.get('scale')).toBe('4')
    expect(form.get('preserveMetadata')).toBe('true')
    expect(form.has('image')).toBe(true)
    expect(form.get('settings')).toBe('{}')
  })
})

describe('createJob settings field', () => {
  it('carries an explicit denoise through untouched', async () => {
    await createJob(request({ settings: { denoiseStrength: 0 } }))

    expect(submitted().get('settings')).toBe('{"denoiseStrength":0}')
  })
})
