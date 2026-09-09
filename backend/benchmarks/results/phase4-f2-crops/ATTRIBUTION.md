# Attribution for the Phase 4 F2 denoise-strength crops

The PNG files in this directory are **derivative works**. Each is a region of
AuraScale's 4x output from a photograph on Wikimedia Commons, produced by
`realesr-general-x4v3` at one of the five denoise strengths compared in
`../phase4_f2_denoise_report.md`.

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

## Model provenance

All crops come from **`realesr-general-x4v3`** (Real-ESRGAN, BSD-3-Clause,
<https://github.com/xinntao/Real-ESRGAN>), weights already in the production
manifest, with its `realesr-general-wdn-x4v3` denoise pair. Checkpoint hashes
are recorded in `../phase4_f2_measurements.json` under
`provenance.checkpoint_sha256`.

## What each file is

| Pattern | Contents |
|---|---|
| `<category>__<region>__dn<strength>.png` | One region at one denoise strength, at 1:1 on the 4x output |
| `blind__<category>__<region>.png` | The five strengths side by side in a **shuffled** order |

The blind strips are deliberately **not** ordered 0.00 → 1.00: a monotone ladder
tells the eye what it is about to see. The panel order for each strip is
recorded in `../phase4_f2_measurements.json` under `experiment: "strips"`, not
drawn on the image, so a reviewer can form a verdict before learning which
panel is which.

Region coordinates, in source pixels, are the `REGIONS` table shared with
Phase 3C and recorded in `../phase3c_measurements.json` under
`provenance.regions`.
