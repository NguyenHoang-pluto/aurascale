import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { HistoryGrid } from '@/features/history/HistoryGrid'
import { ResultViewer } from '@/features/history/ResultViewer'
import type { JobRecord } from '@/types/job'
import { PageContainer } from './PageContainer'

export function HistoryPage() {
  const { t } = useTranslation('history')
  // Which result is open, if any. Local: it is a property of looking at this
  // page, and nothing else in the app has an interest in it.
  const [viewing, setViewing] = useState<JobRecord | null>(null)

  return (
    <PageContainer
      title={t('page.title')}
      description={t('page.description')}
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
