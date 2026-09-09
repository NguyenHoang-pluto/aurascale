# Attribution for the Phase 4 F4 tile-quality crops

The image files in this directory are **derivative works**. Each is a region of
AuraScale's 4x `RealESRGAN_x4plus` output from a photograph on Wikimedia
Commons, produced at one of the three inference tile sizes compared in
`../phase4_f4_tile_quality_report.md`.

| Crop files | Source photograph | Author | Licence |
|---|---|---|---|
| `landscape-detail__*` | [Mountains in snow, Mountain lake, Chola Valley, Nepal, Himalayas](https://commons.wikimedia.org/wiki/File:Mountains_in_snow,_Mountain_lake,_Chola_Valley,_Nepal,_Himalayas.jpg) | Vyacheslav Argenberg | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `portrait-skin__*` | [Headshot Prof Shafi Ahmed](https://commons.wikimedia.org/wiki/File:Headshot_Prof_Shafi_AHmed_001_photo.jpg) | not stated on Commons | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `text-signage__*` | [SE Ankeny Street sign, Portland, Oregon](https://commons.wikimedia.org/wiki/File:SE_Ankeny_Street_sign,_Portland,_Oregon.jpg) | PortlandAppraisalBlog | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `foliage-texture__*` | [A Bamboo Perspective](https://commons.wikimedia.org/wiki/File:A_Bamboo_Perspective.jpg) | Derk29 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `low-light-noise__*` | [Széchenyi Chain Bridge in Budapest at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0) |

`blind__*` strips are montages of the same regions and carry the licence of
their source photograph.

Authorship and licence are recorded as the Commons API reported them when
`benchmarks/fetch_corpus.py` fetched the corpus, not transcribed by hand.

## Note on share-alike

`text-signage__*` and `foliage-texture__*` derive from **CC BY-SA 4.0**
photographs. Share-alike attaches to adapted material, so those files are
offered under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)
rather than under this repository's licence. The others carry the attribution
requirement of CC BY 4.0; the Budapest crops are CC0.

## Model and pipeline provenance

Upscaled by **`RealESRGAN_x4plus`** (Real-ESRGAN, BSD-3-Clause,
<https://github.com/xinntao/Real-ESRGAN>), weights already in the production
manifest, through the production `RealEsrganUpscaler.upscale` path at
`tile_pad = 16`, denoise unset and sharpening 0.0. Unlike F2 and F3 the neural
pass could **not** be shared between arms: tile size is the independent
variable, so each crop comes from its own inference pass.

## What each file is

| Pattern | Contents |
|---|---|
| `<category>__<region>__t###.jpg` | One texture region at one tile size, 1:1 on the 4x output |
| `<category>__seam-t128__t###.png` | 384x512 centred on a joint that is real for tile 128 and tile 64, and is plain image content for tile 256 |
| `<category>__seam-t064__t###.png` | The same for a joint that is real for tile 64 alone |
| `blind__<category>__*` | The three tile sizes side by side in a **shuffled** order |

`t256`, `t128` and `t064` are the requested tile sizes. Every crop's arm was
checked against `UpscaleReport.tile_size` before it was written, because the
production path silently reduces the tile when free VRAM is short — that is not
a hypothetical, it happened during this phase and is written up in § 4 of the
report.

## Why the seam crops are PNG and the texture crops are not

The texture crops are visual-only and are high-quality JPEG (quality 95, no
chroma subsampling) to keep the artifact set small.

The seam crops are **lossless PNG**, deliberately. JPEG's transform blocks fall
on multiples of 8; every seam coordinate measured here is a multiple of 256, so
block edges would land exactly on the joint under inspection. A compression
artifact appearing precisely where the artifact being measured would appear is
the one confound worth spending disk space to avoid.

## Blind ordering

The blind strips are **not** ordered 256 → 128 → 64. A tile ladder has an
obvious expected direction, and F3 recorded — against its author — that knowing
the direction is enough to make a reader describe a progression the pixels do
not contain. Panel order per strip is recorded in
`../phase4_f4_measurements.json` under `experiment: "strips"`, not drawn on the
image.

## Region choice

One texture region per photograph, from the set shared with Phase 3C, plus two
measured seam regions. That is nine images per photograph against F3's thirty:
tiling, if it does anything, does it at coordinates the geometry already names,
so there is no reason to carpet the frame. The seam locations are not chosen by
eye — they are the highest-scoring joint on each grid, and their coordinates are
in the measurements file under `experiment: "seam-crop"`.

No `identity-face` crop is included. The portrait is represented by
`skin-cheek`, and the report does not rest on a face.
