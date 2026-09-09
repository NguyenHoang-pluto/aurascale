import { describe, expect, it } from 'vitest'
import { ALL_TARGETS, planTarget, planTargets, targetDimensions } from './targetPlanning'
import { TARGET_LONG_EDGE } from '@/types/job'

/**
 * Which target presets a loaded image can reach.
 *
 * These mirror `tests/unit/test_resolution_planner.py` case for case, because
 * a client-side check that disagrees with the backend is worse than none: it
 * would grey out something that works, or offer something that will be
 * refused after the upload.
 */

/**
 * A deliberately restricted model list, kept from before 16x existed: it is
 * still a valid set of factors and these cases still describe the rules. The
 * realistic list a 4x model publishes today is `FOUR_X_MODEL` below.
 */
const SCALES = [2, 4, 8]
/** What every 4x model in the registry now offers. */
const FOUR_X_MODEL = [4, 8, 16]
const LIMIT = 200_000_000

describe('target dimensions', () => {
  it('puts the target on the long edge of a landscape image', () => {
    expect(targetDimensions(1600, 900, '4k')).toEqual({ width: 3840, height: 2160 })
  })

  it('puts the target on the long edge of a portrait image', () => {
    expect(targetDimensions(900, 1600, '4k')).toEqual({ width: 2160, height: 3840 })
  })

  it('keeps a square image square', () => {
    expect(targetDimensions(1000, 1000, '2k')).toEqual({ width: 1920, height: 1920 })
  })

  it('does not force an unusual ratio into a standard frame', () => {
    // 4K is 3840 on the long edge, not 3840x2160 for everything.
    expect(targetDimensions(3000, 1000, '4k')).toEqual({ width: 3840, height: 1280 })
  })
})

describe('reachability', () => {
  it('reports a reachable preset with the factor it would use', () => {
    // 1000 px needs 1.92x for 2K, so 2x.
    const plan = planTarget(1000, 800, '2k', SCALES, LIMIT)

    expect(plan.available).toBe(true)
    expect(plan.neuralScale).toBe(2)
    expect(plan.width).toBe(1920)
    expect(plan.reason).toBeUndefined()
  })

  it('refuses a preset the image is already at or beyond', () => {
    expect(planTarget(1920, 1080, '2k', SCALES, LIMIT)).toMatchObject({
      available: false,
      reason: 'alreadyLarger',
    })
    expect(planTarget(4000, 3000, '2k', SCALES, LIMIT)).toMatchObject({
      available: false,
      reason: 'alreadyLarger',
    })
  })

  it('refuses a preset beyond the largest factor the model offers', () => {
    // 400 px would need 19.2x for 8K, and 8x is the ceiling.
    expect(planTarget(400, 300, '8k', SCALES, LIMIT)).toMatchObject({
      available: false,
      reason: 'beyondMaxScale',
    })
  })

  it('refuses a plan whose neural pass would exceed the pixel limit', () => {
    // 3800 px needs 2.02x for 8K, so 4x - and 3800x3400 at 4x is 206.7 MP.
    expect(planTarget(3800, 3400, '8k', SCALES, LIMIT)).toMatchObject({
      available: false,
      reason: 'exceedsPixelLimit',
    })
  })

  it('judges the limit against the intermediate, not the final size', () => {
    // 1900 px needs 2.02x for 4K, so 4x: 51.7 MP intermediate, 13.2 MP result.
    expect(planTarget(1900, 1700, '4k', SCALES, 50_000_000)).toMatchObject({
      available: false,
      reason: 'exceedsPixelLimit',
    })
  })

  it('respects a model that only offers 2x', () => {
    expect(planTarget(1200, 800, '2k', [2], LIMIT).available).toBe(true)
    expect(planTarget(1200, 800, '4k', [2], LIMIT)).toMatchObject({
      available: false,
      reason: 'beyondMaxScale',
    })
  })

  it('treats a model with no factors as reaching nothing', () => {
    expect(planTarget(1000, 1000, '2k', [], LIMIT).available).toBe(false)
  })

  it('refuses degenerate dimensions rather than dividing by them', () => {
    expect(planTarget(0, 0, '2k', SCALES, LIMIT).available).toBe(false)
  })

  it('never proposes a neural size smaller than the target', () => {
    // Otherwise the final step would be an enlargement, which is forbidden.
    for (const source of [400, 700, 1000, 1900, 2600, 3800]) {
      for (const target of ALL_TARGETS) {
        const plan = planTarget(source, source, target, SCALES, LIMIT)
        if (!plan.available) continue
        expect(source * (plan.neuralScale ?? 0)).toBeGreaterThanOrEqual(
          TARGET_LONG_EDGE[target],
        )
      }
    }
  })
})

describe('planning every preset at once', () => {
  it('returns every preset, in ascending order', () => {
    const plans = planTargets(1000, 800, SCALES, LIMIT)

    expect(plans.map((plan) => plan.target)).toEqual(['2k', '4k', '6k', '8k', '16k'])
  })

  it('marks a small image as able to reach the near presets only', () => {
    // 600 px reaches 2K (3.2x -> 4x) and 4K (6.4x -> 8x), but 6K needs 9.6x.
    const plans = planTargets(600, 450, SCALES, LIMIT)
    const byTarget = Object.fromEntries(plans.map((plan) => [plan.target, plan.available]))

    expect(byTarget['2k']).toBe(true)
    expect(byTarget['4k']).toBe(true)
    expect(byTarget['6k']).toBe(false)
    expect(byTarget['8k']).toBe(false)
  })

  it('marks a large image as past the near presets', () => {
    const plans = planTargets(4000, 3000, SCALES, LIMIT)
    const byTarget = Object.fromEntries(plans.map((plan) => [plan.target, plan.reason]))

    expect(byTarget['2k']).toBe('alreadyLarger')
    expect(byTarget['4k']).toBe('alreadyLarger')
    // 4000 px needs 1.44x for 6K, so 2x - 8000x6000 is 48 MP and fits.
    expect(byTarget['6k']).toBeUndefined()
  })

  it('handles portrait and square sources the same way as landscape', () => {
    const landscape = planTargets(1600, 900, SCALES, LIMIT)
    const portrait = planTargets(900, 1600, SCALES, LIMIT)
    const square = planTargets(1600, 1600, SCALES, LIMIT)

    // The long edge is 1600 in all three, so availability matches.
    expect(landscape.map((p) => p.available)).toEqual(portrait.map((p) => p.available))
    expect(landscape.map((p) => p.available)).toEqual(square.map((p) => p.available))
  })
})

/**
 * 16K, against the factors a 4x model actually publishes.
 *
 * These mirror the `sixteen_k` cases in
 * `backend/tests/unit/test_resolution_planner.py`. The name is the trap the
 * whole module exists to avoid: 16K is a destination and 16x is a multiplier,
 * and which factor reaches the destination depends entirely on the source.
 */
describe('16K', () => {
  it('is 15360 px - eight times the base, not sixteen', () => {
    expect(TARGET_LONG_EDGE['16k']).toBe(15360)
    expect(TARGET_LONG_EDGE['16k']).toBe(8 * TARGET_LONG_EDGE['2k'])
  })

  it('is reached from 1080p by an 8x pass, not a 16x one', () => {
    const plan = planTarget(1920, 1080, '16k', FOUR_X_MODEL, LIMIT)

    expect(plan.available).toBe(true)
    expect(plan.neuralScale).toBe(8)
    expect(plan.width).toBe(15360)
    expect(plan.height).toBe(8640)
  })

  it('leaves 8K on the 4x pass it already used', () => {
    // The neighbouring case, unchanged, so the pair reads together.
    const plan = planTarget(1920, 1080, '8k', FOUR_X_MODEL, LIMIT)

    expect(plan.available).toBe(true)
    expect(plan.neuralScale).toBe(4)
    expect(plan.width).toBe(7680)
  })

  it.each([
    // A quarter of 16K: one 4x pass, then a resample down.
    [4000, 2250, 4],
    // An eighth: 8x, landing exactly.
    [1920, 1080, 8],
    // A sixteenth: the only case where 16K really does mean 16x.
    [1000, 563, 16],
  ])('needs %ix%i -> %ix, decided entirely by the source', (width, height, expected) => {
    const plan = planTarget(width, height, '16k', FOUR_X_MODEL, LIMIT)

    expect(plan.neuralScale).toBe(expected)
    expect(Math.max(plan.width ?? 0, plan.height ?? 0)).toBe(15360)
  })

  it('keeps the aspect ratio of a portrait source', () => {
    const plan = planTarget(1080, 1920, '16k', FOUR_X_MODEL, LIMIT)

    expect(plan.width).toBe(8640)
    expect(plan.height).toBe(15360)
    expect(plan.neuralScale).toBe(8)
  })

  it('puts 16K on the long edge whichever edge that is', () => {
    expect(targetDimensions(1920, 1080, '16k')).toEqual({ width: 15360, height: 8640 })
    expect(targetDimensions(1080, 1920, '16k')).toEqual({ width: 8640, height: 15360 })
    expect(targetDimensions(1000, 1000, '16k')).toEqual({ width: 15360, height: 15360 })
  })

  it('refuses a square source where it accepts a wide one', () => {
    // Same long edge, so the same factor - but nearly twice the area, and it
    // is the *neural* size the limit is measured against.
    expect(planTarget(1920, 1080, '16k', FOUR_X_MODEL, LIMIT).available).toBe(true)
    expect(planTarget(1920, 1920, '16k', FOUR_X_MODEL, LIMIT)).toMatchObject({
      available: false,
      reason: 'exceedsPixelLimit',
      neuralScale: 8,
    })
  })

  it('reports the pixel limit against the neural size, not the target', () => {
    // 3000 px needs 5.12x, so 8x - and 24000x16000 is 384 MP even though the
    // 15360x10240 target the user asked for is only 157 MP.
    expect(planTarget(3000, 2000, '16k', FOUR_X_MODEL, LIMIT)).toMatchObject({
      available: false,
      reason: 'exceedsPixelLimit',
      neuralScale: 8,
    })
  })

  it('refuses a source already past 16K', () => {
    expect(planTarget(16000, 9000, '16k', FOUR_X_MODEL, LIMIT)).toMatchObject({
      available: false,
      reason: 'alreadyLarger',
    })
  })

  it('refuses a source too small for any available factor', () => {
    // 400 px would need 38.4x, and nothing offers that.
    expect(planTarget(400, 300, '16k', FOUR_X_MODEL, LIMIT)).toMatchObject({
      available: false,
      reason: 'beyondMaxScale',
    })
  })

  it('is out of reach for a 2x model, which composes nothing', () => {
    expect(planTarget(4000, 2250, '16k', [2], LIMIT)).toMatchObject({
      available: false,
      reason: 'beyondMaxScale',
    })
  })

  it('never proposes a neural size smaller than the 16K target', () => {
    for (const source of [1000, 1200, 1920, 2200, 3900, 4400]) {
      const plan = planTarget(source, Math.round((source * 9) / 16), '16k', FOUR_X_MODEL, LIMIT)
      if (!plan.available) continue
      expect(source * (plan.neuralScale ?? 0)).toBeGreaterThanOrEqual(15360)
    }
  })
})
