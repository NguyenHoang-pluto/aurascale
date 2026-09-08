# Attribution for the comparison crops

The PNG files in this directory are **derivative works**. Each is a montage of
small regions taken from AuraScale's 4x output of a photograph from Wikimedia
Commons, shown at five denoise settings for visual comparison in
`../phase2_5_denoise_report.md`.

The underlying photographs and their licences:

| Crop files | Source photograph | Author | Licence |
|---|---|---|---|
| `v-landscape-detail-*` | [Mountains in snow, Mountain lake, Chola Valley, Nepal, Himalayas](https://commons.wikimedia.org/wiki/File:Mountains_in_snow,_Mountain_lake,_Chola_Valley,_Nepal,_Himalayas.jpg) | Vyacheslav Argenberg | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `v-portrait-skin-*` | [Headshot Prof Shafi Ahmed](https://commons.wikimedia.org/wiki/File:Headshot_Prof_Shafi_AHmed_001_photo.jpg) | not stated on Commons | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `v-text-signage-*` | [SE Ankeny Street sign, Portland, Oregon](https://commons.wikimedia.org/wiki/File:SE_Ankeny_Street_sign,_Portland,_Oregon.jpg) | PortlandAppraisalBlog | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `v-foliage-texture-*` | [Cephalomanes fern foliage close-up, Espiritu Santo, Vanuatu](https://commons.wikimedia.org/wiki/File:(Cephalomanes_fern_foliage_close-up_in_Espiritu_Santo,_Vanuatu)_-_DPLA_-_970b3b811cfb7b136fe34d425d0e9741.jpg) | Sherwin John Carlquist | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) |
| `v-low-light-noise-*` | [Széchenyi Chain Bridge in Budapest at night](https://commons.wikimedia.org/wiki/File:Sz%C3%A9chenyi_Chain_Bridge_in_Budapest_at_night.jpg) | Wilfredor | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0) |

## Note on share-alike

`v-text-signage-*` derives from a **CC BY-SA 4.0** photograph. Share-alike
attaches to adapted material, so those four files are offered under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) rather than
under this repository's licence. The remaining crops require attribution only
(CC BY 4.0) or none (CC0).

If that is unwanted, the `v-text-signage-*` files can be deleted without
affecting the benchmark: they are visual evidence, regenerable from the corpus,
and the numeric results for that image do not depend on them.

The corpus photographs themselves are not stored in this repository. See
`../../corpus/manifest.json` for full provenance and
`benchmarks/fetch_corpus.py` to reproduce them.
