import { Languages } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import {
  SelectContent,
  SelectItem,
  SelectRoot,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { changeLanguage } from '@/i18n'
import { cn } from '@/lib/cn'
import { LANGUAGES, isLanguage, useLanguageStore } from '@/stores/useLanguageStore'

/**
 * The language picker.
 *
 * Each language is named in its own language — "English", "Tiếng Việt" — so
 * someone who cannot read the current interface can still find theirs. That is
 * why the option labels come from a fixed table rather than from `t()`, which
 * would render both of them in whichever language is already active.
 *
 * The switch goes through `changeLanguage`, which moves i18next, the persisted
 * store and `document.lang` together; setting any one of them alone would leave
 * the other two behind.
 */

/** Each language, endonymously. Never translated. */
const ENDONYMS: Record<(typeof LANGUAGES)[number], string> = {
  en: 'English',
  vi: 'Tiếng Việt',
}

export function LanguageSwitcher({
  className,
  compact = false,
  id,
  describedBy,
}: {
  className?: string
  /** Show only the icon and the code, for the nav bar. */
  compact?: boolean
  /** Set when a `Field` labels this control, so the label points at it. */
  id?: string
  describedBy?: string | undefined
}) {
  const { t } = useTranslation('nav')
  const language = useLanguageStore((s) => s.language)

  return (
    <SelectRoot
      value={language}
      onValueChange={(value) => {
        if (isLanguage(value)) changeLanguage(value)
      }}
    >
      <SelectTrigger
        {...(id !== undefined ? { id } : {})}
        {...(describedBy !== undefined ? { 'aria-describedby': describedBy } : {})}
        aria-label={t('language.label')}
        className={cn(compact && 'h-9 w-auto gap-1.5 border-0 bg-transparent px-2', className)}
      >
        {compact ? (
          <>
            <Languages aria-hidden="true" className="size-4 text-muted-foreground" />
            <span className="text-sm">{language.toUpperCase()}</span>
          </>
        ) : (
          <SelectValue />
        )}
      </SelectTrigger>
      <SelectContent>
        {LANGUAGES.map((value) => (
          <SelectItem key={value} value={value}>
            {ENDONYMS[value]}
          </SelectItem>
        ))}
      </SelectContent>
    </SelectRoot>
  )
}
