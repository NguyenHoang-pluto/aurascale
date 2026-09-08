import type { TFunction } from 'i18next'
import type { StatusTone } from '@/components/ui/status'
import type { SystemInfo } from '@/types/system'

export interface StatusPresentation {
  tone: StatusTone
  label: string
}

/**
 * What the top bar says, in priority order (§ 14).
 *
 * Backend reachability is decided first: without it, nothing is known about
 * the GPU, and claiming "CPU mode" when the server is simply down would be
 * wrong. Only once the system report has arrived is a device claim made.
 *
 * Lives apart from the indicator so it can be tested — and reused — without
 * mounting a component. It takes `t` rather than returning a key, because tone
 * and label are one decision and splitting them across the call site would let
 * the two drift apart.
 */
export function describeStatus(
  t: TFunction,
  connection: 'checking' | 'online' | 'offline',
  system: SystemInfo | undefined,
): StatusPresentation {
  if (connection === 'offline') {
    return { tone: 'danger', label: t('system:indicator.offline') }
  }
  if (connection === 'checking') {
    return { tone: 'pending', label: t('system:indicator.checking') }
  }
  if (system === undefined) {
    return { tone: 'pending', label: t('system:indicator.reading') }
  }

  return system.device === 'cuda'
    ? { tone: 'success', label: t('system:indicator.gpu') }
    : { tone: 'warning', label: t('system:indicator.cpu') }
}
