import { useEffect, useRef, useState } from 'react'
import { cropUrl } from '@/services/jobsApi'
import type { Size, Transform } from '@/features/viewer/zoomMath'
import { cropForView, sameCrop } from './comparisonMath'

/**
 * Full-resolution detail for the region on screen, above 100% zoom.
 *
 * The base layer is a preview capped at 4096 px, which is the right trade
 * until the screen asks for more detail than that preview holds. For a result
 * inside the cap that is 1:1; for a 16000 px 8x result the preview carries a
 * quarter of a real pixel per output pixel, so it runs out at 26 % zoom rather
 * than at 100 %. `cropForView` works that threshold out from the output size.
 * This hook fetches the actual pixels for the visible region and nothing else.
 *
 * Four rules keep it from becoming a request firehose:
 *
 *  * nothing is fetched while the preview still has more detail than the
 *    screen can show, which is what `cropForView` decides;
 *  * a pan or zoom restarts a debounce, so dragging produces one request when
 *    the user stops rather than one per frame;
 *  * only one request is in flight, and a newer region supersedes an older
 *    one that has not landed;
 *  * an unchanged region is not refetched.
 *
 * The blob URL is revoked when it is replaced and on unmount, or every crop a
 * long session produces would stay in memory until the tab closed.
 */

export const CROP_DEBOUNCE_MS = 250

export interface CropLayer {
  /** Object URL of the crop, or null when there is nothing to overlay. */
  url: string | null
  /** Where it belongs, in output pixels. */
  region: { x: number; y: number; w: number; h: number } | null
  isLoading: boolean
}

export function useCropLayer(
  jobId: string | null,
  transform: Transform,
  container: Size,
  image: Size,
  options: { enabled?: boolean; debounceMs?: number } = {},
): CropLayer {
  const { enabled = true, debounceMs = CROP_DEBOUNCE_MS } = options

  const [layer, setLayer] = useState<CropLayer>({ url: null, region: null, isLoading: false })

  // Everything the effect needs to compare against without re-subscribing.
  const currentUrl = useRef<string | null>(null)
  const loadedRegion = useRef<CropLayer['region']>(null)
  const inFlight = useRef<AbortController | null>(null)

  const wanted = enabled && jobId !== null ? cropForView(transform, container, image) : null

  // Depended on as four numbers rather than as an object: `cropForView`
  // returns a fresh one every render, which would re-run the effect on every
  // frame of a pan.
  const x = wanted?.x ?? -1
  const y = wanted?.y ?? -1
  const w = wanted?.w ?? 0
  const h = wanted?.h ?? 0

  useEffect(() => {
    const region = w > 0 && h > 0 ? { x, y, w, h } : null

    if (jobId === null || region === null) {
      // Zoomed back out: drop the overlay so the preview is what is on screen,
      // and release the memory with it.
      if (currentUrl.current !== null) {
        URL.revokeObjectURL(currentUrl.current)
        currentUrl.current = null
        loadedRegion.current = null
        setLayer({ url: null, region: null, isLoading: false })
      }
      return
    }

    if (sameCrop(loadedRegion.current, region)) return

    const timer = setTimeout(() => {
      inFlight.current?.abort()
      const controller = new AbortController()
      inFlight.current = controller

      setLayer((previous) => ({ ...previous, isLoading: true }))

      fetch(cropUrl(jobId, region), { signal: controller.signal })
        .then((response) => (response.ok ? response.blob() : Promise.reject(new Error('crop'))))
        .then((blob) => {
          if (controller.signal.aborted) return

          const url = URL.createObjectURL(blob)
          if (currentUrl.current !== null) URL.revokeObjectURL(currentUrl.current)
          currentUrl.current = url
          loadedRegion.current = region
          setLayer({ url, region, isLoading: false })
        })
        .catch(() => {
          if (controller.signal.aborted) return
          // A failed crop is not worth an error panel: the preview underneath
          // is still a correct, if softer, view of the same pixels.
          setLayer((previous) => ({ ...previous, isLoading: false }))
        })
    }, debounceMs)

    return () => { clearTimeout(timer) }
  }, [jobId, x, y, w, h, debounceMs])

  // Release the last crop when the component goes away.
  useEffect(() => {
    return () => {
      inFlight.current?.abort()
      if (currentUrl.current !== null) {
        URL.revokeObjectURL(currentUrl.current)
        currentUrl.current = null
      }
    }
  }, [])

  return layer
}
