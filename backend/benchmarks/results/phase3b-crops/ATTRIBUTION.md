# Attribution for the Phase 3B chroma × DNI crops

The PNG files in this directory are **derivative works**. Each is a montage of
one small region of AuraScale's 4x output from a photograph on Wikimedia
Commons, shown across the four interaction arms compared in
`../phase3b_chroma_dni_report.md`.

Unlike the Phase 3A crops — which showed *filtered inputs* — these are taken
from **super-resolution output**, because Phase 3B's question is what the model
produces once a prefilter and DNI have both acted.

| Crop files | Source photograph | Author | Licence |
|---|---|---|---|
| `p3b-landscape-detail-*` | [Mountains in snow, Mountain lake, Chola Valley, Nepal, Himalayas](https://commons.wikimedia.org/wiki/File:Mountains_in_snow,_Mountain_lake,_Chola_Valley,_Nepal,_Himalayas.jpg) | Vyacheslav Argenberg | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `p3b-portrait-skin-*` | [Headshot Prof Shafi Ahmed](https://commons.wikimedia.org/wiki/File:Headshot_Prof_Shafi_AHmed_001_photo.jpg) | not stated on Commons | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `p3b-text-signage-*` | [SE Ankeny Street sign, Portland, Oregon](https://commons.wikimedia.org/wiki/File:SE_Ankeny_Street_sign,_Portland,_Oregon.jpg) | PortlandAppraisalBlog | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `p3b-foliage-texture-*` | [A Bamboo Perspective](https://commons.wikimedia.org/wiki/File:A_Bamboo_Perspective.jpg) | Derk29 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `p3b-low-light-noise-*` | [Széchenyi Chain Bridge in Budapest at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0) |

Authorship and licence are recorded as the Commons API reported them when
`benchmarks/fetch_corpus.py` fetched the corpus, not transcribed by hand.

## Note on share-alike

`p3b-text-signage-*` and `p3b-foliage-texture-*` derive from **CC BY-SA 4.0**
photographs. Share-alike attaches to adapted material, so those files are
offered under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)
rather than under this repository's licence. The others carry only the
attribution requirement of CC BY 4.0, and the Budapest crops are CC0.

## What each file is

`p3b-<category>-DABC.png` — four panels at 2x nearest-neighbour zoom, in this
order, all from `realesr-general-x4v3` at 4x:

| Panel | Arm | Chroma prefilter | DNI |
|---|---|---|---|
| 1 | **D** baseline | off | 0.00 |
| 2 | **A** DNI only | off | 0.25 |
| 3 | **B** chroma only | `chroma-medium` | 0.00 |
| 4 | **C** chroma + DNI | `chroma-medium` | 0.25 |

Panels are separated by a three-pixel white rule. Crop coordinates are given in
source pixels in `../phase3b_measurements.json` and scaled by 4 into the
output, so every arm shows the same region of the same photograph.

`realesr-general-x4v3` is used for the visual strips because it is the only
registry model with a `denoise_pair`; `RealESRGAN_x4plus` has no DNI, so arms
A and C do not exist for it.
