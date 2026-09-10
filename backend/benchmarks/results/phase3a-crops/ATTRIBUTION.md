# Attribution for the Phase 3A pre-denoise crops

The PNG files in this directory are **derivative works**. Each is a montage of
one small region of a photograph from Wikimedia Commons, shown across several
pre-denoise candidate filters for visual comparison in
`../phase3a_adaptive_denoise_report.md`.

The regions are taken from the **input** photographs, not from AuraScale
output: Phase 3A measures filters that would run *before* super-resolution, so
the comparison that matters is of the filtered input.

| Crop files | Source photograph | Author | Licence |
|---|---|---|---|
| `*-landscape-detail-*` | [Mountains in snow, Mountain lake, Chola Valley, Nepal, Himalayas](https://commons.wikimedia.org/wiki/File:Mountains_in_snow,_Mountain_lake,_Chola_Valley,_Nepal,_Himalayas.jpg) | Vyacheslav Argenberg | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `*-portrait-skin-*` | [Headshot Prof Shafi Ahmed](https://commons.wikimedia.org/wiki/File:Headshot_Prof_Shafi_AHmed_001_photo.jpg) | not stated on Commons | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `*-text-signage-*` | [SE Ankeny Street sign, Portland, Oregon](https://commons.wikimedia.org/wiki/File:SE_Ankeny_Street_sign,_Portland,_Oregon.jpg) | PortlandAppraisalBlog | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `*-foliage-texture-*` | [A Bamboo Perspective](https://commons.wikimedia.org/wiki/File:A_Bamboo_Perspective.jpg) | Derk29 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `*-low-light-noise-*` | [Széchenyi Chain Bridge in Budapest at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0) |

Authorship and licence are recorded as the Commons API reported them when
`benchmarks/fetch_corpus.py` fetched the corpus, not transcribed by hand.

## Note on share-alike

`*-text-signage-*` and `*-foliage-texture-*` derive from **CC BY-SA 4.0**
photographs. Share-alike attaches to adapted material, so those files are
offered under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)
rather than under this repository's licence. The others carry only the
attribution requirement of CC BY 4.0, and the Budapest crops are CC0.

## What each file is

| Suffix | Contents |
|---|---|
| `p3a-<category>-key5.png` | Five panels at 2x: `none`, `gaussian-medium`, `bilateral-medium`, `chroma-medium`, `nlm-medium`. The strip the report's visual observations were made from. |
| `<category>-strip.png` | All fifteen arms at 3x, in `CANDIDATES` order. The full ladder, for checking a specific rung. |

Panels are separated by a two-pixel white rule. Zoom is nearest-neighbour, so
no resampling is introduced between the filter output and the eye.
