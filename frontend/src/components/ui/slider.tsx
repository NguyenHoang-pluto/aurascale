import * as SliderPrimitive from '@radix-ui/react-slider'
import type { ComponentPropsWithoutRef } from 'react'
import { cn } from '@/lib/cn'

/**
 * Single-thumb slider for continuous strengths (sharpening, denoise).
 *
 * Radix provides the keyboard contract § 19 asks for: arrows step, Home/End
 * jump to the bounds, Page Up/Down take larger steps, and the thumb carries
 * aria-valuenow/min/max. Pass `aria-label` or `aria-labelledby` at the call
 * site so the value is announced with a name.
 */
export function Slider({
  className,
  'aria-label': ariaLabel,
  'aria-labelledby': ariaLabelledBy,
  'aria-describedby': ariaDescribedBy,
  ...props
}: ComponentPropsWithoutRef<typeof SliderPrimitive.Root>) {
  return (
    <SliderPrimitive.Root
      className={cn(
        'relative flex w-full touch-none items-center select-none',
        'data-[disabled]:opacity-45',
        className,
      )}
      {...props}
    >
      <SliderPrimitive.Track className="relative h-1 w-full grow overflow-hidden rounded-full bg-input">
        <SliderPrimitive.Range className="absolute h-full bg-accent" />
      </SliderPrimitive.Track>
      <SliderPrimitive.Thumb
        // The naming attributes belong on the thumb, not the root: Radix puts
        // role="slider" there, so a label on the wrapper leaves the control
        // itself anonymous to a screen reader.
        {...(ariaLabel !== undefined ? { 'aria-label': ariaLabel } : {})}
        {...(ariaLabelledBy !== undefined ? { 'aria-labelledby': ariaLabelledBy } : {})}
        {...(ariaDescribedBy !== undefined ? { 'aria-describedby': ariaDescribedBy } : {})}
        className={cn(
          'block size-4 rounded-full border-2 border-accent bg-background',
          'transition-colors hover:bg-muted',
          'disabled:pointer-events-none',
        )}
      />
    </SliderPrimitive.Root>
  )
}
