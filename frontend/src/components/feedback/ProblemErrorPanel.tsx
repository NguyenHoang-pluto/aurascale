import { useErrorMessage } from '@/i18n/errorMessages'
import type { ProblemDetail } from '@/types/api'
import { ErrorPanel } from './ErrorPanel'

/**
 * An error panel for a problem document from the API.
 *
 * The one place a backend failure becomes a sentence. It reads the stable
 * `code` and the `context`, and renders a translated message — the English
 * `detail` the server wrote is never shown as the primary text, because it is
 * written in one language and the user may be reading another.
 *
 * `title` can be overridden where the surrounding screen has better context
 * than the code does ("Could not load history" says more than "Cannot reach
 * the backend"), but the explanation underneath always comes from the code.
 */
export function ProblemErrorPanel({
  problem,
  title,
  onRetry,
  retryLabel,
  className,
}: {
  problem: Pick<ProblemDetail, 'code' | 'detail' | 'context'> & { technical?: string | null }
  title?: string
  onRetry?: () => void
  retryLabel?: string
  className?: string
}) {
  const translate = useErrorMessage()
  const message = translate(problem)

  return (
    <ErrorPanel
      title={title ?? message.title}
      detail={message.detail}
      code={message.code}
      {...(message.technical !== undefined ? { technical: message.technical } : {})}
      {...(onRetry !== undefined ? { onRetry } : {})}
      {...(retryLabel !== undefined ? { retryLabel } : {})}
      {...(className !== undefined ? { className } : {})}
    />
  )
}
