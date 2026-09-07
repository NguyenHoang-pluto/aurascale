# Image processing pipeline

How a file becomes an upscaled image, and why each step exists.

> **Status.** Steps 1-5 of the validation sequence are implemented in the
> browser as of Phase 4 (`frontend/src/lib/imageValidation.ts`), giving
> immediate feedback before an upload is attempted. The backend repeats all of
> them — client checks are convenience, never a security control. Everything
> from tiling onward is the design the Phase 6 implementation is written
> against; the honesty table in §7 records which controls will be real.

## 1. Stages

```
upload
  │
  ├─ 1. size guard          streamed; rejects before buffering the body
  ├─ 2. content sniff       magic bytes; extension and MIME are ignored
  ├─ 3. structural verify   PIL.verify(), then reopen
  ├─ 4. dimension guard     min 32 px, max MAX_INPUT_PIXELS, output projection
  ├─ 5. decode              EXIF orientation applied; ICC and EXIF stashed
  ├─ 6. alpha split         RGBA is separated; alpha never enters the model
  ├─ 7. normalise           uint8 → float32 [0,1] → NCHW → fp16 on CUDA
  │
  ├─ 8. TILED INFERENCE     ◄── the only stage that uses the GPU
  │       for each tile: pad, forward, crop, composite
  │       progress = tilesDone / tilesTotal
  │       cancellation checked between tiles
  │       OOM → halve tile → retry → CPU fallback
  │
  ├─ 9. second pass         only for 8x: the 2x model over the 4x result
  ├─ 10. post-process       unsharp mask if sharpenStrength > 0
  ├─ 11. alpha recomposite  Lanczos-upscaled alpha reattached
  ├─ 12. encode             PNG / JPEG / WEBP; ICC and EXIF reattached
  └─ 13. persist            output, preview and thumbnail written
```

## 2. Validation, in order and why

Order matters: each check is cheaper than the one after it, and each protects
the next from hostile input.

1. **Streamed size cap.** The request body is read in chunks against
   `MAX_UPLOAD_SIZE_MB` and aborted the moment it is exceeded. Checking
   `Content-Length` alone is not enough — it is client-supplied.
2. **Magic bytes.** JPEG (`FF D8 FF`), PNG (`89 50 4E 47`), WEBP (`RIFF....WEBP`).
   The extension and the declared MIME type are attacker-controlled and are not
   consulted for the decision.
3. **`PIL.Image.verify()`.** Catches truncated and structurally corrupt files.
   `verify()` invalidates the file handle, so the image is reopened afterwards —
   a well-known trap.
4. **Dimension guard.** Below `MIN_INPUT_DIMENSION` there is nothing to
   reconstruct. Above `MAX_INPUT_PIXELS` we refuse *before decoding*, which is
   what makes it a decompression-bomb defence rather than a cosmetic limit.
   Pillow's own `MAX_IMAGE_PIXELS` is also left enabled as a backstop.
5. **Output projection.** `width * height * scale²` is checked against
   `MAX_OUTPUT_PIXELS`. A 4000×4000 input at 8× is 1024 MP — refusing at
   submission is far kinder than failing after four minutes of compute.

## 3. Why alpha is handled separately

RRDBNet has three input channels. Feeding an alpha channel through it produces
plausible-looking but invented edges in the transparency mask, which shows up as
halos around cut-outs.

Alpha is therefore split off before inference and upscaled with Lanczos
resampling, then recomposited. Lanczos on a mask is honest: it is a resample,
not a reconstruction, and the UI does not claim otherwise.

## 4. Tiling

A 4× pass on 1280×720 produces 5120×2880. The intermediate activations of a
23-block RRDBNet at that size are far beyond 4 GB of VRAM, so tiling is not an
optimisation here — it is the only way the job runs at all.

**Geometry.** The input is divided into a grid of `tile × tile` cells. Each cell
is expanded by `tile_pad` pixels on every side (clamped at the image edge)
before being fed to the model. After inference the padding is cropped off at
`tile_pad * scale`, and the remainder is written into the output canvas.

The padding is what removes seams. Convolutions near a tile boundary would
otherwise see zero-padding instead of real neighbouring pixels, producing a
visible grid. `tile_pad = 16` gives the receptive field enough real context;
smaller values start to show seams on high-frequency textures.

**Cost.** Padding is redundant compute. With `tile=256, tile_pad=16` each tile
processes 288² instead of 256², about 27% overhead. Larger tiles amortise the
padding better, so tile size is chosen as large as VRAM allows.

**Auto-selection.** At job start, free VRAM is sampled with
`torch.cuda.mem_get_info()` and a tile size is chosen from a conservative table,
clamped by the configured `TILE_SIZE`. The table is calibrated by measurement,
not derived analytically — activation memory depends on the block count, which
varies by model.

**OOM ladder.** `torch.cuda.OutOfMemoryError` is caught per tile:

```
empty_cache() → tile //= 2 → retry   (up to 3 times)
                            ↓ still failing
                      fall back to CPU
```

The job records the tile size actually used, so the UI can report "reduced tile
size to fit available GPU memory" rather than surfacing a CUDA traceback.

**Progress.** `tilesDone / tilesTotal` is a genuine measurement of work
completed. This is the reason the tiling loop is ours rather than
`RealESRGANer.enhance()`, which is opaque.

## 5. Scale factors

| Scale | How | Neural throughout |
| --- | --- | --- |
| 2× | `RealESRGAN_x2plus`, one pass | yes |
| 4× | `RealESRGAN_x4plus`, one pass | yes |
| 8× | `RealESRGAN_x4plus` then `RealESRGAN_x2plus` | yes |

There is no official 8× weight. The alternatives were a 4× pass followed by
bicubic resampling (half of it not AI), or 4× twice to 16× followed by
downsampling (wasteful and soft). The two-pass 4×→2× route keeps every pixel
model-generated, and the UI labels it "two-pass" with a correspondingly larger
time estimate.

## 6. Colour, precision and metadata

- Inference runs in RGB. OpenCV's BGR convention is converted at the boundary,
  never carried into the model.
- fp16 on CUDA, fp32 on CPU. Half precision halves activation memory and is
  faster on every CUDA card that supports it; on CPU it is emulated and slower.
- The output is clamped to `[0,1]` before conversion back to uint8. Without the
  clamp, out-of-range values wrap and produce speckled artefacts.
- ICC profiles are carried through unchanged when metadata is preserved, so a
  Display-P3 image does not silently become sRGB.
- EXIF orientation is applied to the pixels at decode and the tag is normalised,
  so the result is not rotated twice by a viewer that honours it.
- When metadata is stripped, GPS and maker-note tags go first; they are the
  privacy-relevant ones.

## 7. Which controls are real

Requirement §6 asks that no control pretend to be an AI feature it is not. The
commitment for Phase 6:

| Control | Implementation | Honest label |
| --- | --- | --- |
| Model selection | Different trained weights | AI |
| Upscale factor | One or two neural passes | AI |
| Detail enhancement | This *is* the super-resolution model; not a separate toggle | — |
| Noise reduction | DNI interpolation between `realesr-general-x4v3` and its `wdn` counterpart | AI (Phase 6b; "Coming soon" until then) |
| Sharpening | OpenCV unsharp mask on the result | Post-process |
| Artifact reduction | Not implemented | "Coming soon" |

On artifact reduction specifically: doing it properly needs a dedicated
JPEG-restoration model such as FBCNN. A bilateral filter dressed up as
"AI artifact removal" would smear detail while claiming to add it, so the
control ships visibly disabled until the real thing is in place.

### DNI, briefly

Deep Network Interpolation blends two state dicts trained from the same
initialisation:

```
θ_blend = α · θ_standard + (1 − α) · θ_denoise
```

Because `realesr-general-x4v3` and `realesr-general-wdn-x4v3` are exactly such a
pair, the denoise slider maps directly onto α. This is an upstream-supported
technique, not an approximation: at α = 1 you get the standard model's weights
byte for byte, at α = 0 the denoise model's, and the blend in between is a real
network, not a blend of two outputs.

## 8. Memory discipline

- Uploads stream to a temporary file; a full-size image is never accumulated in
  memory just to measure it.
- Only one decoded image and one output canvas are resident per job, which is
  why `MAX_CONCURRENT_JOBS` defaults to 1.
- `torch.inference_mode()` wraps every forward pass, so no autograd graph is
  built.
- `torch.cuda.empty_cache()` runs after each job, keeping the allocator from
  holding fragmented blocks across jobs.
- Temporary files are removed on both the success and failure paths, and a
  periodic sweeper removes anything past `TEMP_RETENTION_HOURS` that earlier
  paths missed.
