import { PhaseNotice } from '@/components/feedback/PhaseNotice'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { SystemPanel } from '@/features/system/SystemPanel'
import { PageContainer } from './PageContainer'

const PENDING_SECTIONS = [
  {
    title: 'General',
    description: 'Theme and workspace preferences.',
    phase: 'Phase 14',
  },
  {
    title: 'Processing',
    description: 'Default model, default scale, tile size and tile padding.',
    phase: 'Phase 8',
  },
  {
    title: 'Storage',
    description: 'Temporary file retention and maximum upload size.',
    phase: 'Phase 10',
  },
] as const

export function SettingsPage() {
  return (
    <PageContainer
      title="Settings"
      description="Defaults for processing, storage and the local backend."
    >
      <div className="flex flex-col gap-4">
        {PENDING_SECTIONS.map((section) => (
          <Panel key={section.title}>
            <PanelHeader>
              <PanelTitle>{section.title}</PanelTitle>
            </PanelHeader>
            <PanelContent>
              <PhaseNotice
                title={section.title}
                description={section.description}
                phase={section.phase}
                className="border-0 px-0 py-4"
              />
            </PanelContent>
          </Panel>
        ))}

        {/* Live, measured on the server — not a placeholder. */}
        <Panel>
          <PanelHeader>
            <PanelTitle>System</PanelTitle>
          </PanelHeader>
          <PanelContent>
            <SystemPanel />
          </PanelContent>
        </Panel>
      </div>
    </PageContainer>
  )
}
