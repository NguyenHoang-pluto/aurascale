import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { App } from './App'
import { initTheme } from './stores/useThemeStore'
import './styles/index.css'

initTheme()

const rootElement = document.getElementById('root')
if (rootElement === null) {
  throw new Error('Root element #root not found in index.html')
}

createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
