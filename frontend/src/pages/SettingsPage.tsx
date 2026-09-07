import { SlidersHorizontal } from 'lucide-react'
import { PlaceholderPage } from './PlaceholderPage'

export function SettingsPage() {
  return (
    <PlaceholderPage
      icon={SlidersHorizontal}
      title="Settings"
      description="Default model and scale, tile size and padding, storage retention, and live backend/GPU status."
      phase="Phase 5"
    />
  )
}
