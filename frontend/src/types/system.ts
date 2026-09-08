/**
 * System and model types.
 *
 * These mirror the response schemas in `backend/app/schemas/system.py` and
 * `model.py`. The backend serialises camelCase, so no renaming happens here.
 */

export type DeviceType = 'cuda' | 'cpu'

export interface TorchStatus {
  available: boolean
  version: string | null
  cudaVersion: string | null
  cudaAvailable: boolean
  importError: string | null
}

export interface GpuStatus {
  name: string
  vramTotalMb: number
  vramFreeMb: number
  /** CUDA compute capability, e.g. "8.6". */
  capability: string
}

export interface SystemInfo {
  device: DeviceType
  /** Why that device was selected, shown verbatim in Settings. */
  deviceReason: string
  torch: TorchStatus
  /** Null when CUDA is unavailable — a normal state, not an error. */
  gpu: GpuStatus | null
  cpuName: string
  cpuCoresPhysical: number | null
  cpuCoresLogical: number | null
  ramTotalMb: number
  ramAvailableMb: number
  pythonVersion: string
  platform: string
  fp16: boolean
  tileSize: number
  tilePad: number
}

export interface ModelInfo {
  id: string
  name: string
  description: string
  /** Network architecture, e.g. "RRDBNet". */
  arch: string
  /** Native upscale factor of these weights. */
  scale: number
  supportsDenoise: boolean
  /**
   * Upscale factors this model can actually produce, published by the backend
   * so the UI never offers a combination a job would refuse.
   */
  supportedScales: number[]
  downloaded: boolean
  sizeMb: number | null
}
