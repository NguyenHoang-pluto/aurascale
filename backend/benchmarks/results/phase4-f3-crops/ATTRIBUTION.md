# Attribution for the Phase 4 F3 sharpening crops

The PNG files in this directory are **derivative works**. Each is a region of
AuraScale's 4x `RealESRGAN_x4plus` output from a photograph on Wikimedia
Commons, post-processed at one of the six sharpening strengths compared in
`../phase4_f3_sharpening_report.md`.

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

## Model and post-process provenance

Upscaled by **`RealESRGAN_x4plus`** (Real-ESRGAN, BSD-3-Clause,
<https://github.com/xinntao/Real-ESRGAN>), weights already in the production
manifest, then sharpened by AuraScale's own `unsharp_mask` at
sigma 3.0, dead zone ±2.0, ceiling ±10.0. The neural pass ran **once per
photograph** and all six strengths were applied to that identical array, so the
crops of one region differ only by sharpening strength.

## What each file is

| Pattern | Contents |
|---|---|
| `<category>__<region>__sh<strength>.png` | One region at one sharpening strength, 1:1 on the 4x output |
| `blind__<category>__<region>.png` | The six strengths side by side in a **shuffled** order |

The blind strips are deliberately **not** ordered 0.00 → 0.75. A monotone ladder
tells the eye which way "more" is, and with sharpening that is precisely the
bias to avoid, since the question under test is whether more is better. Panel
order per strip is recorded in `../phase4_f3_measurements.json` under
`experiment: "strips"`, not drawn on the image.

Ten regions are the set shared with Phase 3C. Two are specific to F3 —
`landscape-detail/snow-rock-edge` and `low-light-noise/lamp-edge` — chosen
because sharpening's characteristic failure is at strong edges and neither
existing region was a maximally hard boundary. Coordinates for all twelve are
in the measurements file under `provenance.regions`.
