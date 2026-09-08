import { useTranslation } from 'react-i18next'
import { StatusIndicator } from '@/components/ui/status'
import { Tooltip } from '@/components/ui/tooltip'
import { formatDuration, formatNumber } from '@/lib/format'
import type { SystemInfo } from '@/types/system'
import { describeStatus } from './describeStatus'
import { useBackendHealth } from './useBackendHealth'
import { useSystemInfo } from './useSystemInfo'

/** GB is a unit symbol, identical in both languages; only the number moves. */
function formatGb(megabytes: number, locale: string): string {
  return `${formatNumber(megabytes / 1024, locale, 1)} GB`
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
  const { t, i18n } = useTranslation(['system', 'common'])
  const locale = i18n.language

  if (connection === 'offline') {
    return (
      <div className="space-y-0.5">
        <p className="font-medium">{t('indicator.offline')}</p>
        <p className="text-muted-foreground">{reason ?? t('indicator.offlineHint')}</p>
        <p className="mt-1 text-muted-foreground">{t('indicator.startHint')}</p>
      </div>
    )
  }

  if (system === undefined) {
    return <p>{t('indicator.contacting')}</p>
  }

  return (
    <div className="space-y-0.5">
      <p className="font-medium">
        {system.device === 'cuda' ? t('indicator.gpu') : t('indicator.cpu')}
      </p>
      <p className="text-muted-foreground">{system.deviceReason}</p>

      {system.gpu !== null ? (
        <>
          <p className="text-muted-foreground">{system.gpu.name}</p>
          <p className="text-muted-foreground">
            {t('indicator.vram', {
              free: formatGb(system.gpu.vramFreeMb, locale),
              total: formatGb(system.gpu.vramTotalMb, locale),
            })}
          </p>
        </>
      ) : (
        <p className="text-muted-foreground">
          {system.cpuName}
          {system.cpuCoresLogical !== null &&
            ` · ${t('indicator.threads', { threads: system.cpuCoresLogical })}`}
        </p>
      )}

      <p className="text-muted-foreground">
        {/* "torch", "CUDA" and "fp16" are product names, never translated. */}
        torch {system.torch.version ?? t('common:state.unknown')}
        {system.torch.cudaVersion !== null && ` · CUDA ${system.torch.cudaVersion}`}
        {system.fp16 && ' · fp16'}
      </p>

      {version !== undefined && uptimeSeconds !== undefined && (
        <p className="mt-1 text-muted-foreground">
          {t('indicator.uptime', {
            version,
            uptime: formatDuration(uptimeSeconds * 1000, locale),
          })}
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
  const { t } = useTranslation(['system', 'common'])
  const { state, data: health, reason } = useBackendHealth()
  const { data: system } = useSystemInfo()

  const { tone, label } = describeStatus(t, state, system)

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
