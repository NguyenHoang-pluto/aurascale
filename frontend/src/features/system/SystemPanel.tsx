import { RefreshCw } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { StatusIndicator } from '@/components/ui/status'
import { formatNumber } from '@/lib/format'
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

/** GB is a unit symbol, identical in both languages; only the number moves. */
function gb(megabytes: number, locale: string): string {
  return `${formatNumber(megabytes / 1024, locale, 1)} GB`
}

function DeviceSummary({ system }: { system: SystemInfo }) {
  const { t } = useTranslation('system')

  return (
    <div className="flex flex-wrap items-center gap-3">
      {system.device === 'cuda' ? (
        <StatusIndicator tone="success" label={t('indicator.gpu')} />
      ) : (
        <StatusIndicator tone="warning" label={t('indicator.cpu')} />
      )}
      <span className="text-xs text-muted-foreground">{system.deviceReason}</span>
    </div>
  )
}

function ModelList({ models }: { models: ModelInfo[] }) {
  const { t } = useTranslation('system')

  return (
    <ul className="flex flex-col gap-2">
      {models.map((model) => (
        <li key={model.id} className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-xs font-medium text-foreground">{model.name}</p>
            <p className="truncate font-mono text-[11px] text-muted-foreground">
              {/* Architecture names and the factor are identifiers. */}
              {model.arch} · {model.scale}x
              {model.supportsDenoise && ` · ${t('panel.denoiseSuffix')}`}
            </p>
          </div>
          {model.downloaded ? (
            <Badge tone="success">
              {model.sizeMb !== null ? `${String(model.sizeMb)} MB` : t('panel.downloaded')}
            </Badge>
          ) : (
            <Badge tone="neutral">{t('panel.notDownloaded')}</Badge>
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
  const { t, i18n } = useTranslation(['system', 'common'])
  const locale = i18n.language
  const { data: system, isPending, isError, reason, refetch } = useSystemInfo()
  const models = useModels()

  if (isError) {
    return (
      <ErrorPanel
        title={t('system:panel.loadFailed')}
        detail={reason ?? t('system:panel.loadFailedDetail')}
        code="backend_unavailable"
        technical={t('system:panel.loadFailedTechnical')}
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
        <Button
          variant="ghost"
          size="sm"
          onClick={refetch}
          aria-label={t('system:panel.refresh')}
        >
          <RefreshCw aria-hidden="true" />
          {t('common:actions.refresh')}
        </Button>
      </div>

      <Separator />

      <dl>
        {system.gpu !== null ? (
          <>
            <Row label={t('system:panel.gpu')} value={system.gpu.name} />
            <Row
              label={t('system:panel.vram')}
              value={t('system:panel.vramValue', {
                free: gb(system.gpu.vramFreeMb, locale),
                total: gb(system.gpu.vramTotalMb, locale),
              })}
            />
            <Row label={t('system:panel.capability')} value={system.gpu.capability} />
          </>
        ) : (
          <Row label={t('system:panel.gpu')} value={t('system:panel.noGpu')} />
        )}

        <Row
          label={t('system:panel.cudaAvailable')}
          value={system.torch.cudaAvailable ? t('common:state.yes') : t('common:state.no')}
        />
        <Row
          label={t('system:panel.cudaVersion')}
          value={system.torch.cudaVersion ?? t('common:state.none')}
        />
        <Row
          label={t('system:panel.torch')}
          value={system.torch.version ?? t('system:panel.notAvailable')}
        />
        <Row
          label={t('system:panel.fp16')}
          value={system.fp16 ? t('common:state.enabled') : t('common:state.disabled')}
        />

        <Separator className="my-2" />

        <Row label={t('system:panel.cpu')} value={system.cpuName} />
        <Row
          label={t('system:panel.cores')}
          value={
            system.cpuCoresPhysical !== null && system.cpuCoresLogical !== null
              ? t('system:panel.coresValue', {
                  physical: system.cpuCoresPhysical,
                  logical: system.cpuCoresLogical,
                })
              : t('common:state.none')
          }
        />
        <Row
          label={t('system:panel.memory')}
          value={t('system:panel.memoryValue', {
            available: gb(system.ramAvailableMb, locale),
            total: gb(system.ramTotalMb, locale),
          })}
        />
        <Row label={t('system:panel.platform')} value={system.platform} />
        <Row label={t('system:panel.python')} value={system.pythonVersion} />

        <Separator className="my-2" />

        <Row
          label={t('system:panel.tileSize')}
          value={
            <>
              <MetricBadge>{system.tileSize}px</MetricBadge>{' '}
              <span className="text-muted-foreground">
                {t('system:panel.tilePad', { pad: system.tilePad })}
              </span>
            </>
          }
        />
      </dl>

      <Separator />

      <div>
        <p className="mb-2 text-xs font-medium text-foreground">{t('system:panel.models')}</p>
        {models.isError ? (
          <p className="text-xs text-muted-foreground">
            {models.reason ?? t('system:panel.modelsFailed')}
          </p>
        ) : models.data === undefined ? (
          <Skeleton className="h-8 w-full" />
        ) : (
          <>
            <ModelList models={models.data} />
            {models.data.every((model) => !model.downloaded) && (
              <p className="mt-2 text-xs text-muted-foreground">
                {t('system:panel.weightsHint')}
              </p>
            )}
          </>
        )}
      </div>
    </div>
  )
}
