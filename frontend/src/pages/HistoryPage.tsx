import { useState } from 'react'
import { HistoryGrid } from '@/features/history/HistoryGrid'
import { ResultViewer } from '@/features/history/ResultViewer'
import type { JobRecord } from '@/types/job'
import { PageContainer } from './PageContainer'

export function HistoryPage() {
  // Which result is open, if any. Local: it is a property of looking at this
  // page, and nothing else in the app has an interest in it.
  const [viewing, setViewing] = useState<JobRecord | null>(null)

  return (
    <PageContainer
      title="History"
      description="Recently enhanced images, with the model, scale and processing time used for each."
    >
      <div className="flex flex-col gap-4">
        {viewing !== null && (
          <ResultViewer job={viewing} onClose={() => { setViewing(null) }} />
        )}
        <HistoryGrid onView={setViewing} />
      </div>
    </PageContainer>
  )
}
