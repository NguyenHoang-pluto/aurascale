import { StatusIndicator } from '@/components/ui/status'
import { Tooltip } from '@/components/ui/tooltip'
import { formatDuration } from '@/lib/format'
import type { SystemInfo } from '@/types/system'
import { describeStatus } from './describeStatus'
import { useBackendHealth } from './useBackendHealth'
import { useSystemInfo } from './useSystemInfo'

function formatGb(megabytes: number): string {
  return `${(megabytes / 1024).toFixed(1)} GB`
}

function StatusTooltip({
  connection,
  system,
  reason,
  version,
  uptimeSeconds,
}: {
  connection: 'checking' | 'online' | 'offline'
  system: SystemInfo | undefined
  reason: string | undefined
  version: string | undefined
  uptimeSeconds: number | undefined
}) {
  if (connection === 'offline') {
    return (
      <div className="space-y-0.5">
        <p className="font-medium">Backend offline</p>
        <p className="text-muted-foreground">{reason ?? 'The backend is not responding.'}</p>
        <p className="mt-1 text-muted-foreground">Start it with scripts/dev.ps1</p>
      </div>
    )
  }

  if (system === undefined) {
    return <p>Contacting the backend…</p>
  }

  return (
    <div className="space-y-0.5">
      <p className="font-medium">
        {system.device === 'cuda' ? 'GPU acceleration enabled' : 'CPU mode'}
      </p>
      <p className="text-muted-foreground">{system.deviceReason}</p>

      {system.gpu !== null ? (
        <>
          <p className="text-muted-foreground">{system.gpu.name}</p>
          <p className="text-muted-foreground">
            {formatGb(system.gpu.vramFreeMb)} free of {formatGb(system.gpu.vramTotalMb)} VRAM
          </p>
        </>
      ) : (
        <p className="text-muted-foreground">
          {system.cpuName}
          {system.cpuCoresLogical !== null && ` · ${String(system.cpuCoresLogical)} threads`}
        </p>
      )}

      <p className="text-muted-foreground">
        torch {system.torch.version ?? 'unknown'}
        {system.torch.cudaVersion !== null && ` · CUDA ${system.torch.cudaVersion}`}
        {system.fp16 && ' · fp16'}
      </p>

      {version !== undefined && uptimeSeconds !== undefined && (
        <p className="mt-1 text-muted-foreground">
          v{version} · up {formatDuration(uptimeSeconds * 1000)}
        </p>
      )}
    </div>
  )
}

/**
 * Top-bar status (§ 4, § 14).
 *
 * Reports what is measured on the server, never a guess: "GPU acceleration
 * enabled" only appears once `/api/system` has confirmed a usable CUDA device.
 */
export function BackendStatusIndicator() {
  const { state, data: health, reason } = useBackendHealth()
  const { data: system } = useSystemInfo()

  const { tone, label } = describeStatus(state, system)

  return (
    <Tooltip
      content={
        <StatusTooltip
          connection={state}
          system={system}
          reason={reason}
          version={health?.version}
          uptimeSeconds={health?.uptimeSeconds}
        />
      }
    >
      <span className="hidden rounded-md border border-border bg-surface-raised px-2.5 py-1 sm:inline-flex">
        <StatusIndicator live tone={tone} label={label} />
      </span>
    </Tooltip>
  )
}
