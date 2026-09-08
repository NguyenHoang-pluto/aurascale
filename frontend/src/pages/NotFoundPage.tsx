import { FileQuestion } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { NavLink } from 'react-router-dom'
import { EmptyState } from '@/components/feedback/EmptyState'
import { Button } from '@/components/ui/button'

export function NotFoundPage() {
  const { t } = useTranslation('common')

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center">
      <EmptyState
        icon={FileQuestion}
        title={t('notFound.title')}
        description={t('notFound.description')}
        action={
          <Button variant="secondary" asChild>
            <NavLink to="/">{t('notFound.back')}</NavLink>
          </Button>
        }
      />
    </div>
  )
}
