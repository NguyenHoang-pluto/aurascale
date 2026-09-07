/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL for the backend REST API. Empty string means "same origin" (dev proxy / nginx). */
  readonly VITE_API_BASE_URL?: string
  /** App version surfaced in the settings page. */
  readonly VITE_APP_VERSION?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
