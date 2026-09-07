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
 * mounting a component.
 */
export function describeStatus(
  connection: 'checking' | 'online' | 'offline',
  system: SystemInfo | undefined,
): StatusPresentation {
  if (connection === 'offline') return { tone: 'danger', label: 'Backend offline' }
  if (connection === 'checking') return { tone: 'pending', label: 'Checking backend' }
  if (system === undefined) return { tone: 'pending', label: 'Reading capabilities' }

  return system.device === 'cuda'
    ? { tone: 'success', label: 'GPU acceleration enabled' }
    : { tone: 'warning', label: 'CPU mode' }
}
