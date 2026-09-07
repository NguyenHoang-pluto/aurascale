import { StatusIndicator, type StatusTone } from '@/components/ui/status'
import { Tooltip } from '@/components/ui/tooltip'
import { formatDuration } from '@/lib/format'
import { useBackendHealth, type BackendConnectionState } from './useBackendHealth'

const TONE_BY_STATE: Record<BackendConnectionState, StatusTone> = {
  checking: 'pending',
  online: 'success',
  offline: 'danger',
}

const LABEL_BY_STATE: Record<BackendConnectionState, string> = {
  checking: 'Checking backend',
  online: 'Backend online',
  offline: 'Backend offline',
}

/**
 * Top-bar backend status.
 *
 * § 4 places a GPU status indicator here. GPU and CUDA state come from
 * `/api/system`, which is a Phase 5 endpoint, so this reports the reachability
 * it can actually verify — and says nothing it cannot. Phase 5 extends this
 * component to show "GPU acceleration enabled" or "CPU mode".
 */
export function BackendStatusIndicator() {
  const { state, data, reason } = useBackendHealth()

  const tooltip =
    state === 'online' && data !== undefined ? (
      <div className="space-y-0.5">
        <p className="font-medium">Backend online</p>
        <p className="text-muted-foreground">
          v{data.version} · {data.environment}
        </p>
        <p className="text-muted-foreground">
          Uptime {formatDuration(data.uptimeSeconds * 1000)}
        </p>
        <p className="mt-1 text-muted-foreground">GPU status arrives in Phase 5.</p>
      </div>
    ) : state === 'offline' ? (
      <div className="space-y-0.5">
        <p className="font-medium">Backend offline</p>
        <p className="text-muted-foreground">{reason ?? 'The backend is not responding.'}</p>
        <p className="mt-1 text-muted-foreground">Start it with scripts/dev.ps1</p>
      </div>
    ) : (
      <p>Contacting the backend…</p>
    )

  return (
    <Tooltip content={tooltip}>
      <span className="hidden rounded-md border border-border bg-surface-raised px-2.5 py-1 sm:inline-flex">
        <StatusIndicator live tone={TONE_BY_STATE[state]} label={LABEL_BY_STATE[state]} />
      </span>
    </Tooltip>
  )
}
