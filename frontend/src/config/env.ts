/**
 * Typed access to build-time environment. Centralised so no component reads
 * `import.meta.env` directly and every default lives in one place.
 */
export interface AppEnv {
  /** Base URL prefix for API calls. '' = same origin, proxied in dev by Vite. */
  readonly apiBaseUrl: string
  readonly appVersion: string
  readonly isDev: boolean
}

function readEnv(): AppEnv {
  return {
    apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? '',
    appVersion: import.meta.env.VITE_APP_VERSION ?? '0.1.0',
    isDev: import.meta.env.DEV,
  }
}

export const env: AppEnv = readEnv()
