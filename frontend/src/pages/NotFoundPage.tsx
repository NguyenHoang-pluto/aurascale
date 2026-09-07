import { NavLink } from 'react-router-dom'
import { FileQuestion } from 'lucide-react'
import { EmptyState } from '@/components/feedback/EmptyState'
import { Button } from '@/components/ui/button'

export function NotFoundPage() {
  return (
    <div className="flex min-h-0 flex-1 items-center justify-center">
      <EmptyState
        icon={FileQuestion}
        title="Page not found"
        description="That route does not exist."
        action={
          <Button variant="secondary" asChild>
            <NavLink to="/">Back to workspace</NavLink>
          </Button>
        }
      />
    </div>
  )
}
