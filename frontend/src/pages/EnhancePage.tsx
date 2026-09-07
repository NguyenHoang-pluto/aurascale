import { ImageUp } from 'lucide-react'
import { PlaceholderPage } from './PlaceholderPage'

export function EnhancePage() {
  return (
    <PlaceholderPage
      icon={ImageUp}
      title="Enhance"
      description="Upload, compare and upscale images. The upload area and image viewer are built in Phase 4; the comparison viewer in Phase 9."
      phase="Phase 4"
    />
  )
}
