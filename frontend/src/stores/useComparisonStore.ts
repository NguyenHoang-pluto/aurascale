import { create } from 'zustand'
import { SPLIT_POSITION, clampDivider, type CompareMode } from '@/features/compare/comparisonMath'

/**
 * Comparison mode and divider position.
 *
 * Client state, per the layering rule in docs/architecture.md § 11: nothing on
 * the server has an opinion about how the user is looking at their result.
 *
 * The "before" snapshot lives here too. It is taken from the `File` at the
 * moment a job is submitted and owns its own object URL, so replacing the
 * workspace image mid-job cannot change what the comparison is comparing —
 * and cannot leave this store holding a URL another store has revoked.
 */

export interface BeforeSnapshot {
  objectUrl: string
  width: number
  height: number
  name: string
}

interface ComparisonState {
  mode: CompareMode
  /** 0-100. Meaningful in slider mode; split is fixed and side-by-side ignores it. */
  divider: number
  before: BeforeSnapshot | null

  setMode: (mode: CompareMode) => void
  setDivider: (position: number) => void
  /** Take a snapshot of the submitted file, replacing and revoking any previous one. */
  captureBefore: (file: File, size: { width: number; height: number }) => void
  clearBefore: () => void
}

function revoke(snapshot: BeforeSnapshot | null): void {
  if (snapshot !== null) URL.revokeObjectURL(snapshot.objectUrl)
}

export const useComparisonStore = create<ComparisonState>()((set, get) => ({
  mode: 'slider',
  divider: SPLIT_POSITION,
  before: null,

  setMode: (mode) => { set({ mode }) },
  setDivider: (position) => { set({ divider: clampDivider(position) }) },

  captureBefore: (file, size) => {
    revoke(get().before)
    set({
      before: {
        objectUrl: URL.createObjectURL(file),
        width: size.width,
        height: size.height,
        name: file.name,
      },
    })
  },

  clearBefore: () => {
    revoke(get().before)
    set({ before: null })
  },
}))
