import { RefreshCw } from 'lucide-react'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { StatusIndicator } from '@/components/ui/status'
import type { ModelInfo, SystemInfo } from '@/types/system'
import { useModels, useSystemInfo } from './useSystemInfo'

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="text-right font-mono text-xs tabular-nums text-foreground">{value}</dd>
    </div>
  )
}

function gb(megabytes: number): string {
  return `${(megabytes / 1024).toFixed(1)} GB`
}

function DeviceSummary({ system }: { system: SystemInfo }) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      {system.device === 'cuda' ? (
        <StatusIndicator tone="success" label="GPU acceleration enabled" />
      ) : (
        <StatusIndicator tone="warning" label="CPU mode" />
      )}
      <span className="text-xs text-muted-foreground">{system.deviceReason}</span>
    </div>
  )
}

function ModelList({ models }: { models: ModelInfo[] }) {
  return (
    <ul className="flex flex-col gap-2">
      {models.map((model) => (
        <li key={model.id} className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-xs font-medium text-foreground">{model.name}</p>
            <p className="truncate font-mono text-[11px] text-muted-foreground">
              {model.arch} · {model.scale}x{model.supportsDenoise && ' · denoise'}
            </p>
          </div>
          {model.downloaded ? (
            <Badge tone="success">
              {model.sizeMb !== null ? `${String(model.sizeMb)} MB` : 'Downloaded'}
            </Badge>
          ) : (
            <Badge tone="neutral">Not downloaded</Badge>
          )}
        </li>
      ))}
    </ul>
  )
}

/**
 * The System section of Settings (§ 21).
 *
 * Every value is read from `GET /api/system` and `GET /api/models` — the same
 * measurements the inference pipeline will act on, so what is displayed here
 * is what will actually happen.
 */
export function SystemPanel() {
  const { data: system, isPending, isError, reason, refetch } = useSystemInfo()
  const models = useModels()

  if (isError) {
    return (
      <ErrorPanel
        title="Cannot reach the backend"
        detail={reason ?? 'The backend is not responding.'}
        code="backend_unavailable"
        technical="GET /api/system failed. Start the backend with scripts/dev.ps1 or ./scripts/dev.sh"
        onRetry={refetch}
      />
    )
  }

  if (isPending || system === undefined) {
    return (
      <div className="flex flex-col gap-2" aria-busy="true">
        <Skeleton className="h-4 w-56" />
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-4 w-48" />
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-start justify-between gap-3">
        <DeviceSummary system={system} />
        <Button variant="ghost" size="sm" onClick={refetch} aria-label="Refresh system status">
          <RefreshCw aria-hidden="true" />
          Refresh
        </Button>
      </div>

      <Separator />

      <dl>
        {system.gpu !== null ? (
          <>
            <Row label="GPU" value={system.gpu.name} />
            <Row
              label="VRAM"
              value={`${gb(system.gpu.vramFreeMb)} free / ${gb(system.gpu.vramTotalMb)}`}
            />
            <Row label="Compute capability" value={system.gpu.capability} />
          </>
        ) : (
          <Row label="GPU" value="None detected" />
        )}

        <Row label="CUDA available" value={system.torch.cudaAvailable ? 'Yes' : 'No'} />
        <Row label="CUDA version" value={system.torch.cudaVersion ?? '—'} />
        <Row label="PyTorch" value={system.torch.version ?? 'Not available'} />
        <Row label="Half precision (fp16)" value={system.fp16 ? 'Enabled' : 'Disabled'} />

        <Separator className="my-2" />

        <Row label="CPU" value={system.cpuName} />
        <Row
          label="Cores"
          value={
            system.cpuCoresPhysical !== null && system.cpuCoresLogical !== null
              ? `${String(system.cpuCoresPhysical)} physical / ${String(system.cpuCoresLogical)} logical`
              : '—'
          }
        />
        <Row
          label="Memory"
          value={`${gb(system.ramAvailableMb)} available / ${gb(system.ramTotalMb)}`}
        />
        <Row label="Platform" value={system.platform} />
        <Row label="Python" value={system.pythonVersion} />

        <Separator className="my-2" />

        <Row
          label="Tile size"
          value={
            <>
              <MetricBadge>{system.tileSize}px</MetricBadge>{' '}
              <span className="text-muted-foreground">pad {system.tilePad}px</span>
            </>
          }
        />
      </dl>

      <Separator />

      <div>
        <p className="mb-2 text-xs font-medium text-foreground">Models</p>
        {models.isError ? (
          <p className="text-xs text-muted-foreground">
            {models.reason ?? 'The model list could not be loaded.'}
          </p>
        ) : models.data === undefined ? (
          <Skeleton className="h-8 w-full" />
        ) : (
          <>
            <ModelList models={models.data} />
            {models.data.every((model) => !model.downloaded) && (
              <p className="mt-2 text-xs text-muted-foreground">
                Weights are downloaded on first use, or ahead of time with
                scripts/download_models.py (Phase 6).
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
