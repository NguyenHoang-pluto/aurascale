import { Slot } from '@radix-ui/react-slot'
import { cva, type VariantProps } from 'class-variance-authority'
import type { ComponentPropsWithoutRef } from 'react'
import { cn } from '@/lib/cn'

const buttonVariants = cva(
  [
    'inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-md',
    'text-sm font-medium transition-colors duration-150',
    'disabled:pointer-events-none disabled:opacity-45',
    // Icons inside buttons are sized by the button, never by the caller.
    '[&_svg]:size-4 [&_svg]:shrink-0',
  ],
  {
    variants: {
      variant: {
        /** The single most important action on a screen. Use sparingly. */
        primary: 'bg-accent text-accent-foreground hover:bg-accent/90 active:bg-accent/80',
        secondary:
          'border border-border bg-surface-raised text-foreground hover:bg-muted active:bg-muted',
        outline: 'border border-border bg-transparent text-foreground hover:bg-muted',
        ghost: 'text-muted-foreground hover:bg-muted hover:text-foreground',
        destructive:
          'bg-destructive text-destructive-foreground hover:bg-destructive/90 active:bg-destructive/80',
      },
      size: {
        sm: 'h-8 px-2.5 text-xs',
        md: 'h-9 px-3.5',
        lg: 'h-11 px-6 text-base',
        'icon-sm': 'size-8',
        icon: 'size-9',
      },
    },
    defaultVariants: { variant: 'secondary', size: 'md' },
  },
)

export interface ButtonProps
  extends ComponentPropsWithoutRef<'button'>,
    VariantProps<typeof buttonVariants> {
  /**
   * Render the child element instead of a `<button>`, forwarding styles onto it.
   * Use for links that should look like buttons, so the accessible role stays
   * correct rather than putting a click handler on a div.
   */
  asChild?: boolean
}

export function Button({
  className,
  variant,
  size,
  asChild = false,
  type,
  ...props
}: ButtonProps) {
  const Component = asChild ? Slot : 'button'

  return (
    <Component
      // A <button> inside a form defaults to type="submit", which submits
      // unexpectedly. Only set it when we are actually rendering a button.
      {...(asChild ? {} : { type: type ?? 'button' })}
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  )
}
