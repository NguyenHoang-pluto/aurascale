# Attribution for the Phase 3C detail-model crops

The PNG files in this directory are **derivative works**. Each is a region of
AuraScale's 4x output from a photograph on Wikimedia Commons, produced by one of
the four models compared in `../phase3c_detail_model_report.md`.

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
`benchmarks/fetch_corpus.py` fetched the corpus.

## Note on share-alike

`text-signage__*` and `foliage-texture__*` derive from **CC BY-SA 4.0**
photographs, so those files are offered under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0). The others carry
the attribution requirement of CC BY 4.0; the Budapest crops are CC0.

## Model provenance

Three arms come from **Real-ESRGAN** (BSD-3-Clause,
<https://github.com/xinntao/Real-ESRGAN>), using weights already in the
production manifest.

The `hat` arm is **Real_HAT_GAN_SRx4** from **HAT** (Apache-2.0,
<https://github.com/XPixelGroup/HAT>), `sha256
f5b1e3bbbb05147ca2beefcc715279cb647d7976cbda67d62ea7e6e20d5ffcc7`, obtained
from a third-party HuggingFace mirror because the official release is Google
Drive only; the hash was verified against an independent listing after
download. The architecture is vendored, research-only, at
`benchmarks/arch/hat.py` and is **never imported by `app/`**.

## What each file is

| Pattern | Contents |
|---|---|
| `<category>__<region>__<arm>.png` | One region of one arm's 4x output, at 1:1 |
| `blind__<category>__<region>.png` | The four arms side by side in a **shuffled** order; the mapping is in `../phase3c_measurements.json` under `experiment: "strips"`, not drawn on the image |

Panels are separated by a white rule. Region coordinates, in source pixels, are
recorded in the measurements file under `provenance.regions`.
