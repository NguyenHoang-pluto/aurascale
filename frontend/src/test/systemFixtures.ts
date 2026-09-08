/**
 * Shared fixtures for the system endpoints.
 *
 * `BackendStatusIndicator` is mounted by the top bar on every screen, so any
 * test that renders the shell answers `/api/system` too. Keeping one copy of
 * the payloads here stops those answers from drifting apart.
 */

import { vi } from 'vitest'
import { jsonResponse } from './renderWithProviders'
import type { HealthResponse } from '@/types/api'
import type { ModelInfo, SystemInfo } from '@/types/system'

export const HEALTH: HealthResponse = {
  status: 'ok',
  version: '0.1.0',
  environment: 'test',
  uptimeSeconds: 42,
}

export const GPU_SYSTEM: SystemInfo = {
  device: 'cuda',
  deviceReason: 'CUDA device detected',
  torch: {
    available: true,
    version: '2.7.1+cu118',
    cudaVersion: '11.8',
    cudaAvailable: true,
    importError: null,
  },
  gpu: {
    name: 'NVIDIA GeForce RTX 3050 Laptop GPU',
    vramTotalMb: 4095,
    vramFreeMb: 3333,
    capability: '8.6',
  },
  cpuName: 'AMD Ryzen 5 5625U with Radeon Graphics',
  cpuCoresPhysical: 6,
  cpuCoresLogical: 12,
  ramTotalMb: 7532,
  ramAvailableMb: 676,
  pythonVersion: '3.11.9',
  platform: 'Windows 10',
  fp16: true,
  tileSize: 256,
  tilePad: 16,
}

export const CPU_SYSTEM: SystemInfo = {
  ...GPU_SYSTEM,
  device: 'cpu',
  deviceReason: 'no CUDA device available',
  torch: { ...GPU_SYSTEM.torch, cudaVersion: null, cudaAvailable: false },
  gpu: null,
  fp16: false,
}

export const MODELS: ModelInfo[] = [
  {
    id: 'RealESRGAN_x4plus',
    name: 'Real-ESRGAN x4 Plus',
    description: 'General-purpose 4x upscaler.',
    arch: 'RRDBNet',
    scale: 4,
    supportsDenoise: false,
    supportedScales: [4, 8],
    downloaded: false,
    sizeMb: null,
  },
  {
    id: 'realesr-general-x4v3',
    name: 'Real-ESRGAN General v3',
    description: 'Compact 4x model.',
    arch: 'SRVGGNetCompact',
    scale: 4,
    supportsDenoise: true,
    supportedScales: [4, 8],
    downloaded: true,
    sizeMb: 4.7,
  },
]

/**
 * Route each endpoint to a body, so components exercise the real API client.
 *
 * A request to an unlisted path rejects the way an unreachable backend does,
 * which keeps a test from passing on a payload it never meant to serve.
 */
export function stubApi(routes: Record<string, unknown>, status = 200) {
  const fetchMock = vi.fn((url: string) => {
    const match = Object.keys(routes).find((path) => url.includes(path))
    if (match === undefined) return Promise.reject(new TypeError('Failed to fetch'))
    return Promise.resolve(jsonResponse(routes[match], status))
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

/** Every system endpoint answered, for tests that only render the shell. */
export function stubSystemApi(system: SystemInfo = GPU_SYSTEM) {
  return stubApi({ '/api/health': HEALTH, '/api/system': system, '/api/models': MODELS })
}
