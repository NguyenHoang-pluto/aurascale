import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { TooltipProvider } from '@/components/ui/tooltip'
import { DesignSystemPage } from '@/pages/DesignSystemPage'
import { EnhancePage } from '@/pages/EnhancePage'
import { HistoryPage } from '@/pages/HistoryPage'
import { NotFoundPage } from '@/pages/NotFoundPage'
import { SettingsPage } from '@/pages/SettingsPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Job and system state is polled explicitly; avoid surprise refetches
      // that would fight the SSE stream added in Phase 7.
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5_000,
    },
  },
})

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      {/* delayDuration keeps tooltips from flashing during pointer transit. */}
      <TooltipProvider delayDuration={300} skipDelayDuration={150}>
        <BrowserRouter>
          <AppShell>
            <Routes>
              <Route path="/" element={<EnhancePage />} />
              <Route path="/history" element={<HistoryPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              {/*
                Component gallery, development builds only.
                `import.meta.env.DEV` rather than our env module: Vite
                substitutes the literal at build time, so the condition folds
                to false and Rollup drops DesignSystemPage from the bundle
                entirely. Reading it through a runtime object would leave the
                page bundled but unreachable.
              */}
              {import.meta.env.DEV && (
                <Route path="/design" element={<DesignSystemPage />} />
              )}
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </AppShell>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  )
}
