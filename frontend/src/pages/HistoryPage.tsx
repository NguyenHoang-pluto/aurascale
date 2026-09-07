import { Clock } from 'lucide-react'
import { PlaceholderPage } from './PlaceholderPage'

export function HistoryPage() {
  return (
    <PlaceholderPage
      icon={Clock}
      title="History"
      description="Previously enhanced images with their model, scale and processing time. Backed by SQLite via the job repository."
      phase="Phase 10"
    />
  )
}
