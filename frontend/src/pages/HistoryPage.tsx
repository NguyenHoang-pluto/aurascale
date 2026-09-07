import { PhaseNotice } from '@/components/feedback/PhaseNotice'
import { PageContainer } from './PageContainer'

export function HistoryPage() {
  return (
    <PageContainer
      title="History"
      description="Previously enhanced images, with the model, scale and processing time used for each."
    >
      <PhaseNotice
        title="Enhancement history"
        description="Thumbnails, input and output dimensions, scale, model, processing time and creation date, with view, download and delete actions. Backed by SQLite through the job repository."
        phase="Phase 10"
      />
    </PageContainer>
  )
}
