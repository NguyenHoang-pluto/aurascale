/**
 * Stubs for browser APIs jsdom does not implement.
 *
 * The object-URL stub tracks creations and revocations so tests can assert the
 * workspace store does not leak blobs — a leak is invisible in a unit test
 * otherwise, and only shows up as growing memory in a long session.
 */

const created = new Set<string>()
const revoked = new Set<string>()
let counter = 0

export function installObjectUrl(): void {
  URL.createObjectURL = ((): string => {
    counter += 1
    const url = `blob:pixelforge/${String(counter)}`
    created.add(url)
    return url
  }) as typeof URL.createObjectURL

  URL.revokeObjectURL = ((url: string): void => {
    revoked.add(url)
  }) as typeof URL.revokeObjectURL
}

export function resetObjectUrl(): void {
  created.clear()
  revoked.clear()
  counter = 0
}

/** Object URLs created but never revoked. */
export function leakedObjectUrls(): string[] {
  return [...created].filter((url) => !revoked.has(url))
}

export function createdObjectUrlCount(): number {
  return created.size
}

type ResizeCallback = (entries: ResizeObserverEntry[]) => void

const resizeSubscriptions = new Map<Element, Set<ResizeCallback>>()

/**
 * A ResizeObserver that tests can actually drive.
 *
 * jsdom has no ResizeObserver at all, and a no-op stub would leave any
 * size-dependent component permanently believing it has zero space -- which
 * silently skips the code path being tested. This one records observers so
 * `resizeElement` can deliver a real entry, exercising the production path.
 */
export function installResizeObserver(): void {
  globalThis.ResizeObserver = class {
    private readonly callback: ResizeCallback
    private readonly observed = new Set<Element>()

    constructor(callback: ResizeCallback) {
      this.callback = callback
    }

    observe(element: Element): void {
      this.observed.add(element)
      let callbacks = resizeSubscriptions.get(element)
      if (callbacks === undefined) {
        callbacks = new Set()
        resizeSubscriptions.set(element, callbacks)
      }
      callbacks.add(this.callback)
    }

    unobserve(element: Element): void {
      this.observed.delete(element)
      resizeSubscriptions.get(element)?.delete(this.callback)
    }

    disconnect(): void {
      for (const element of this.observed) {
        resizeSubscriptions.get(element)?.delete(this.callback)
      }
      this.observed.clear()
    }
  } as unknown as typeof ResizeObserver
}

export function resetResizeObserver(): void {
  resizeSubscriptions.clear()
}

/**
 * Give an element a size and notify its observers, as a real layout change
 * would. Callers must wrap this in `act()` so React flushes the update.
 */
export function resizeElement(element: Element, width: number, height: number): void {
  stubBoundingRect(element, width, height)

  const entry = {
    target: element,
    contentRect: { width, height, top: 0, left: 0, right: width, bottom: height, x: 0, y: 0 },
  } as unknown as ResizeObserverEntry

  for (const callback of resizeSubscriptions.get(element) ?? []) {
    callback([entry])
  }
}

/**
 * Give an element a non-zero layout box. jsdom reports every rect as zero,
 * which would make the viewer believe it has no space to fit an image into.
 */
export function stubBoundingRect(element: Element, width: number, height: number): void {
  element.getBoundingClientRect = (() =>
    ({
      width,
      height,
      top: 0,
      left: 0,
      right: width,
      bottom: height,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    }) as DOMRect) as Element['getBoundingClientRect']
}
