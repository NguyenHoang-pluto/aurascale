import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { initLanguage } from './i18n'
import { initTheme } from './stores/useThemeStore'
import './styles/index.css'

// Both applied before render so the first painted frame is already in the
// right theme and the right language, rather than flashing and correcting.
initTheme()
initLanguage()

const rootElement = document.getElementById('root')
if (rootElement === null) {
  throw new Error('Root element #root not found in index.html')
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
