import { create } from 'zustand'
import { validateImageFile, type ImageDecoder } from '@/lib/imageValidation'
import type { ProblemDetail } from '@/types/api'
import type { LoadedImage } from '@/types/image'

/**
 * The image currently loaded into the workspace.
 *
 * Not persisted: a `File` and its object URL are only meaningful for the life
 * of the document, and writing one to storage would be a lie on reload.
 *
 * This store owns every object URL it creates and revokes the previous one on
 * every transition. Without that, each replaced image leaks its full encoded
 * size for as long as the tab is open.
 */
interface WorkspaceState {
  source: LoadedImage | null
  /** Validation or load failure, in the same shape as an API error. */
  problem: ProblemDetail | null
  isLoading: boolean

  loadFile: (file: File, options?: { decode?: ImageDecoder }) => Promise<void>
  clear: () => void
  dismissProblem: () => void
}

function revoke(image: LoadedImage | null): void {
  if (image !== null) URL.revokeObjectURL(image.objectUrl)
}

export const useWorkspaceStore = create<WorkspaceState>()((set, get) => ({
  source: null,
  problem: null,
  isLoading: false,

  loadFile: async (file, options = {}) => {
    set({ isLoading: true, problem: null })

    const result = await validateImageFile(file, {
      ...(options.decode !== undefined ? { decode: options.decode } : {}),
    })

    if (!result.ok) {
      // A rejected file leaves any previously loaded image untouched — losing
      // the current image because the next one was invalid would be hostile.
      set({ isLoading: false, problem: result.problem })
      return
    }

    revoke(get().source)
    set({
      source: {
        file,
        objectUrl: URL.createObjectURL(file),
        metadata: result.metadata,
      },
      problem: null,
      isLoading: false,
    })
  },

  clear: () => {
    revoke(get().source)
    set({ source: null, problem: null, isLoading: false })
  },

  dismissProblem: () => { set({ problem: null }) },
}))
