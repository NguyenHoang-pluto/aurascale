import { useTranslation } from 'react-i18next'
import { PhaseNotice } from '@/components/feedback/PhaseNotice'
import { LanguageSwitcher } from '@/components/layout/LanguageSwitcher'
import { Field } from '@/components/ui/field'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { Separator } from '@/components/ui/separator'
import { SystemPanel } from '@/features/system/SystemPanel'
import { PageContainer } from './PageContainer'

/**
 * The sections whose settings are not built yet.
 *
 * General is not among them any more: language is a real preference now, and
 * it is set here as well as in the top bar, because Settings is where someone
 * looks for it. What the section still lacks is named underneath rather than
 * hidden, so the panel does not imply it is finished.
 */
const PENDING_SECTIONS = [
  {
    id: 'processing',
    title: 'settings.processing',
    description: 'settings.processingDescription',
    phase: 'Phase 8',
  },
  {
    id: 'storage',
    title: 'settings.storage',
    description: 'settings.storageDescription',
    phase: 'Phase 10',
  },
] as const

export function SettingsPage() {
  const { t } = useTranslation(['system', 'nav'])

  return (
    <PageContainer
      title={t('system:settings.title')}
      description={t('system:settings.description')}
    >
      <div className="flex flex-col gap-4">
        <Panel>
          <PanelHeader>
            <PanelTitle>{t('system:settings.general')}</PanelTitle>
          </PanelHeader>
          <PanelContent>
            <Field
              label={t('nav:language.label')}
              description={t('system:settings.languageDescription')}
              orientation="horizontal"
            >
              {({ id, describedBy }) => (
                <LanguageSwitcher id={id} describedBy={describedBy} className="w-44" />
              )}
            </Field>

            <Separator className="my-4" />

            <PhaseNotice
              title={t('system:settings.general')}
              description={t('system:settings.generalDescription')}
              phase="Phase 14"
              className="border-0 px-0 py-4"
            />
          </PanelContent>
        </Panel>

        {PENDING_SECTIONS.map((section) => (
          <Panel key={section.id}>
            <PanelHeader>
              <PanelTitle>{t(`system:${section.title}`)}</PanelTitle>
            </PanelHeader>
            <PanelContent>
              <PhaseNotice
                title={t(`system:${section.title}`)}
                description={t(`system:${section.description}`)}
                phase={section.phase}
                className="border-0 px-0 py-4"
              />
            </PanelContent>
          </Panel>
        ))}

        {/* Live, measured on the server — not a placeholder. */}
        <Panel>
          <PanelHeader>
            <PanelTitle>{t('system:panel.title')}</PanelTitle>
          </PanelHeader>
          <PanelContent>
            <SystemPanel />
          </PanelContent>
        </Panel>
      </div>
    </PageContainer>
  )
}
