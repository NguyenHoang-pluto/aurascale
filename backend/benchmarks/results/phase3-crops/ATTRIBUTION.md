# Attribution for the Phase 3 model-comparison crops

The PNG files in this directory are **derivative works**. Each is a montage of
one small region taken from AuraScale's 4x output of a photograph from
Wikimedia Commons, shown across four model arms for visual comparison in
`../phase3_model_report.md`.

| Crop files | Source photograph | Author | Licence |
|---|---|---|---|
| `p3-landscape-detail-*` | [Mountains in snow, Mountain lake, Chola Valley, Nepal, Himalayas](https://commons.wikimedia.org/wiki/File:Mountains_in_snow,_Mountain_lake,_Chola_Valley,_Nepal,_Himalayas.jpg) | Vyacheslav Argenberg | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `p3-portrait-skin-*` | [Headshot Prof Shafi Ahmed](https://commons.wikimedia.org/wiki/File:Headshot_Prof_Shafi_AHmed_001_photo.jpg) | not stated on Commons | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `p3-text-signage-*` | [SE Ankeny Street sign, Portland, Oregon](https://commons.wikimedia.org/wiki/File:SE_Ankeny_Street_sign,_Portland,_Oregon.jpg) | PortlandAppraisalBlog | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `p3-foliage-texture-*` | [A Bamboo Perspective](https://commons.wikimedia.org/wiki/File:A_Bamboo_Perspective.jpg) | Derk29 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `p3-low-light-noise-*` | [Széchenyi Chain Bridge in Budapest at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0) |

## Note on share-alike

`p3-text-signage-*` and `p3-foliage-texture-*` derive from **CC BY-SA 4.0**
photographs. Share-alike attaches to adapted material, so those six files are
offered under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0)
rather than under this repository's licence. The rest require attribution only
(CC BY 4.0) or none (CC0).

The foliage source changed in Phase 3: the Phase 2.5 crops in `../crops/`
derive from a different, CC BY 4.0 fern photograph, and that directory's own
`ATTRIBUTION.md` still describes them correctly.

If share-alike is unwanted, these six files can be deleted without affecting
the benchmark — they are visual evidence, regenerable from the corpus, and the
numeric results do not depend on them.

`regions.json` records the crop coordinates, in output pixels, so the montages
can be reproduced exactly.

The corpus photographs themselves are not stored in this repository. See
`../../corpus/manifest.json` for full provenance and `benchmarks/fetch_corpus.py`
to reproduce them.
