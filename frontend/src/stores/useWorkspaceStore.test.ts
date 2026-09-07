import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useWorkspaceStore } from './useWorkspaceStore'
import { createdObjectUrlCount, leakedObjectUrls } from '@/test/browserStubs'

const PNG_HEADER = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
const GIF_HEADER = [0x47, 0x49, 0x46, 0x38, 0x39, 0x61]

function makeFile(header: readonly number[], name: string): File {
  const bytes = new Uint8Array(64)
  bytes.set(header)
  return new File([bytes], name, { type: 'image/png' })
}

const decodeAs = (width: number, height: number) => vi.fn(() => Promise.resolve({ width, height }))

describe('useWorkspaceStore', () => {
  beforeEach(() => {
    useWorkspaceStore.setState({ source: null, problem: null, isLoading: false })
  })

  it('loads a valid image and exposes its metadata', async () => {
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(PNG_HEADER, 'a.png'), { decode: decodeAs(1280, 720) })

    const { source, problem, isLoading } = useWorkspaceStore.getState()
    expect(problem).toBeNull()
    expect(isLoading).toBe(false)
    expect(source?.metadata.width).toBe(1280)
    expect(source?.metadata.height).toBe(720)
    expect(source?.objectUrl).toMatch(/^blob:/)
  })

  it('records a validation failure without creating an object URL', async () => {
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(GIF_HEADER, 'a.gif'), { decode: decodeAs(100, 100) })

    const { source, problem } = useWorkspaceStore.getState()
    expect(source).toBeNull()
    expect(problem?.code).toBe('unsupported_format')
    expect(createdObjectUrlCount()).toBe(0)
  })

  it('keeps the current image when a replacement is rejected', async () => {
    const store = useWorkspaceStore.getState()
    await store.loadFile(makeFile(PNG_HEADER, 'good.png'), { decode: decodeAs(800, 600) })
    const original = useWorkspaceStore.getState().source

    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(GIF_HEADER, 'bad.gif'), { decode: decodeAs(800, 600) })

    // Losing a valid image because the next pick was invalid would be hostile.
    expect(useWorkspaceStore.getState().source).toBe(original)
    expect(useWorkspaceStore.getState().problem?.code).toBe('unsupported_format')
  })

  it('revokes the previous object URL when an image is replaced', async () => {
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(PNG_HEADER, 'first.png'), { decode: decodeAs(800, 600) })
    const first = useWorkspaceStore.getState().source?.objectUrl

    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(PNG_HEADER, 'second.png'), { decode: decodeAs(640, 480) })
    const second = useWorkspaceStore.getState().source?.objectUrl

    expect(second).not.toBe(first)
    expect(leakedObjectUrls()).toEqual([second])
  })

  it('revokes the object URL on clear, leaving nothing behind', async () => {
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(PNG_HEADER, 'a.png'), { decode: decodeAs(800, 600) })

    useWorkspaceStore.getState().clear()

    expect(useWorkspaceStore.getState().source).toBeNull()
    expect(leakedObjectUrls()).toEqual([])
  })

  it('clears a previous problem when a new load starts', async () => {
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(GIF_HEADER, 'bad.gif'), { decode: decodeAs(10, 10) })
    expect(useWorkspaceStore.getState().problem).not.toBeNull()

    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(PNG_HEADER, 'good.png'), { decode: decodeAs(800, 600) })

    expect(useWorkspaceStore.getState().problem).toBeNull()
  })

  it('dismisses a problem without discarding the loaded image', async () => {
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(PNG_HEADER, 'a.png'), { decode: decodeAs(800, 600) })
    await useWorkspaceStore
      .getState()
      .loadFile(makeFile(GIF_HEADER, 'b.gif'), { decode: decodeAs(10, 10) })

    useWorkspaceStore.getState().dismissProblem()

    expect(useWorkspaceStore.getState().problem).toBeNull()
    expect(useWorkspaceStore.getState().source).not.toBeNull()
  })
})
