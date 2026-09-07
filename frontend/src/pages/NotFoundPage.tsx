import { FileQuestion } from 'lucide-react'
import { PlaceholderPage } from './PlaceholderPage'

export function NotFoundPage() {
  return (
    <PlaceholderPage
      icon={FileQuestion}
      title="Page not found"
      description="That route does not exist. Use the navigation above to return to the workspace."
      phase="404"
    />
  )
}
