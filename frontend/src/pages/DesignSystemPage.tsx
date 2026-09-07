import { Download, Sparkles, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { ErrorPanel } from '@/components/feedback/ErrorPanel'
import { Badge, MetricBadge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Field } from '@/components/ui/field'
import { Panel, PanelContent, PanelHeader, PanelTitle } from '@/components/ui/panel'
import { Progress } from '@/components/ui/progress'
import { SegmentedControl } from '@/components/ui/segmented-control'
import {
  SelectContent,
  SelectItem,
  SelectRoot,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Skeleton } from '@/components/ui/skeleton'
import { Slider } from '@/components/ui/slider'
import { StatusIndicator } from '@/components/ui/status'
import { Switch } from '@/components/ui/switch'
import { PageContainer } from './PageContainer'

/**
 * Component gallery for visual review of the design system.
 *
 * Development-only: the route is not registered in a production build. The
 * controls here are a live demonstration of the primitives, not product
 * features — nothing on this page talks to the backend.
 */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Panel>
      <PanelHeader>
        <PanelTitle>{title}</PanelTitle>
      </PanelHeader>
      <PanelContent className="flex flex-col gap-4">{children}</PanelContent>
    </Panel>
  )
}

export function DesignSystemPage() {
  const [scale, setScale] = useState<'2' | '4' | '8'>('4')
  const [model, setModel] = useState('RealESRGAN_x4plus')
  const [sharpen, setSharpen] = useState([30])
  const [preserveMetadata, setPreserveMetadata] = useState(true)

  return (
    <PageContainer
      title="Design system"
      description="Development-only gallery of the shared primitives. Not part of a production build."
    >
      <div className="flex flex-col gap-4">
        <Section title="Buttons">
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="primary">
              <Sparkles aria-hidden="true" />
              Enhance image
            </Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="destructive">
              <Trash2 aria-hidden="true" />
              Delete
            </Button>
            <Button variant="secondary" disabled>
              Disabled
            </Button>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm">Small</Button>
            <Button size="md">Medium</Button>
            <Button size="lg">Large</Button>
            <Button size="icon" variant="ghost" aria-label="Download">
              <Download aria-hidden="true" />
            </Button>
          </div>
        </Section>

        <Section title="Status and badges">
          <div className="flex flex-wrap items-center gap-4">
            <StatusIndicator tone="idle" label="Idle" />
            <StatusIndicator tone="pending" label="Running inference" />
            <StatusIndicator tone="success" label="GPU acceleration enabled" />
            <StatusIndicator tone="warning" label="CPU mode" />
            <StatusIndicator tone="danger" label="Backend offline" />
          </div>
          <Separator />
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="neutral">Neutral</Badge>
            <Badge tone="accent">4x</Badge>
            <Badge tone="success">Completed</Badge>
            <Badge tone="warning">Queued</Badge>
            <Badge tone="danger">Failed</Badge>
            <MetricBadge>5120 × 2880</MetricBadge>
            <MetricBadge>8.7 MB</MetricBadge>
          </div>
        </Section>

        <Section title="Form controls">
          <SegmentedControl
            name="demo-scale"
            label="Upscale factor"
            value={scale}
            onChange={setScale}
            options={[
              { value: '2', label: '2x' },
              { value: '4', label: '4x' },
              { value: '8', label: '8x' },
            ]}
          />

          <Field label="Model" description="Which trained weights to run.">
            {({ id, describedBy }) => (
              <SelectRoot value={model} onValueChange={setModel}>
                <SelectTrigger id={id} aria-describedby={describedBy}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="RealESRGAN_x4plus" description="General photographs">
                    Real-ESRGAN x4 Plus
                  </SelectItem>
                  <SelectItem
                    value="RealESRGAN_x4plus_anime_6B"
                    description="Illustration and line art"
                  >
                    Real-ESRGAN x4 Plus Anime
                  </SelectItem>
                </SelectContent>
              </SelectRoot>
            )}
          </Field>

          <Field
            label="Sharpening"
            description="Unsharp mask applied after upscaling. A post-process, not a model feature."
          >
            {({ id, describedBy }) => (
              <div className="flex items-center gap-3">
                <Slider
                  id={id}
                  aria-describedby={describedBy}
                  aria-label="Sharpening strength"
                  value={sharpen}
                  onValueChange={setSharpen}
                  max={100}
                  step={1}
                />
                <MetricBadge className="w-12 justify-center">{sharpen[0] ?? 0}%</MetricBadge>
              </div>
            )}
          </Field>

          <Field
            label="Preserve metadata"
            description="Keep EXIF and the ICC colour profile in the output."
            orientation="horizontal"
          >
            {({ id, describedBy }) => (
              <Switch
                id={id}
                aria-describedby={describedBy}
                checked={preserveMetadata}
                onCheckedChange={setPreserveMetadata}
              />
            )}
          </Field>

          <Field
            label="Artifact reduction"
            description="Requires a dedicated JPEG restoration model."
            orientation="horizontal"
            comingSoon
          >
            {({ id, describedBy }) => (
              <Switch id={id} aria-describedby={describedBy} checked={false} disabled />
            )}
          </Field>
        </Section>

        <Section title="Progress and loading">
          <Progress value={62} label="Demonstration progress" />
          <div className="flex flex-col gap-2">
            <Skeleton className="h-4 w-48" />
            <Skeleton className="h-4 w-32" />
          </div>
        </Section>

        <Section title="Errors">
          <ErrorPanel
            title="Not enough memory"
            detail="Your image is too large to process with the available GPU memory. Try a lower upscale factor, or a smaller tile size in Settings."
            code="out_of_memory"
            technical={
              'RuntimeError: CUDA out of memory. Tried to allocate 1.24 GiB\n' +
              '(GPU 0; 4.00 GiB total capacity; 2.91 GiB already allocated)'
            }
            onRetry={() => undefined}
          />
        </Section>
      </div>
    </PageContainer>
  )
}
