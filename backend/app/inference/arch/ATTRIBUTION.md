# Vendored model architectures

This directory contains neural network architecture definitions copied from
upstream open-source projects. They are vendored rather than installed as
dependencies — see `docs/architecture.md` §6 for the full reasoning.

## Why these files are here and not a pip dependency

The `realesrgan` PyPI package depends on `basicsr`, which imports
`torchvision.transforms.functional_tensor` (removed in torchvision 0.17) and
assumes NumPy < 2. Neither has had a release since 2022, so installing them
alongside a current PyTorch fails at import time. The circulating workarounds
require monkey-patching a third-party package at runtime.

Only the architecture definitions are needed — plain `nn.Module` classes with no
dependency on the rest of `basicsr`. Vendoring them removes the broken
dependency chain entirely while keeping the pretrained weights, and therefore
the output, identical to upstream.

## Contents

| File | Upstream | License | Modifications |
| --- | --- | --- | --- |
| `rrdbnet.py` | [BasicSR](https://github.com/XPixelGroup/BasicSR) `basicsr/archs/rrdbnet_arch.py` | Apache-2.0 | Registry decorator and unused imports removed; type hints added; formatting only |
| `srvggnet.py` | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) `realesrgan/archs/srvgg_arch.py` | BSD-3-Clause | Registry decorator removed; type hints added; formatting only |

**No layer definitions, tensor operations, or numeric behaviour are changed.**
Weights are loaded with `strict=True`, which would fail loudly if any parameter
name or shape had drifted from the released checkpoints — this is asserted by a
test in `tests/integration/`.

These files are excluded from ruff's naming rules and from mypy, so they stay
diffable against upstream.

## Upstream notices

### BasicSR — Apache License 2.0

```
Copyright 2018-2022 BasicSR Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Real-ESRGAN — BSD 3-Clause License

```
Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Model weights

Pretrained weights are **not** vendored. They are downloaded at runtime from the
official Real-ESRGAN release URLs recorded in `models/manifest.json`, verified
against SHA-256 digests, and stored in the git-ignored `models/` directory.
They remain under the Real-ESRGAN project's BSD-3-Clause license.

> The architecture files themselves arrive in Phase 6, together with the
> download-and-verify implementation. This notice is committed ahead of them so
> the licensing position is established before any third-party code lands.
