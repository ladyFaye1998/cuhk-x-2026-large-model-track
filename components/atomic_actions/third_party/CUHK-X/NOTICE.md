# CUHK-X native model source attribution

Copyright (c) 2025–2026 The Chinese University of Hong Kong (AIoT Lab).
All rights reserved.

The native atomic-action model uses three files from the official CUHK-X
repository tree `df03910a6960db5af9179e370e130bf61d9616d0`. Obtain them from
the official source under the CUHK-X terms or use the organizer verification
package. Their upstream locations and identities are:

| File | Official Git blob SHA-1 |
|---|---|
| `SM/Code/skeleton/model/DSTformer.py` | `f7efc52713c46a7dc8685174fb3d9240a20141dd` |
| `SM/Code/skeleton/model/drop.py` | `b79c5e0cb970b5e222a416609c67998c9300d531` |
| `SM/Code/skeleton/model/model_action.py` | `43eb16fcee9a412148d97bd06b6d3ab2654e7b51` |

The original inference layout uses `DSTformer.py`, `drop.py`, and
`model_action.py` under `components/atomic_actions/source/`. The training
layout also uses the three files under
`components/atomic_actions/source/upstream/downstream/`. The organizer's
inference copies differ from the pinned source only in trailing whitespace.
Use the original bundle when verifying packaged byte hashes.

The files are available from the [pinned upstream directory](https://github.com/openaiotlab/CUHK-X/tree/df03910a6960db5af9179e370e130bf61d9616d0/SM/Code/skeleton/model).

The accompanying `LICENSE.txt` is CUHK-X License Version 2.0, retrieved on
2026-09-12. Its SHA-256 is
`f91f3c1aee1491a919a88cf2cdcd247532ef0d21b2b412882163ccac5ea06a03`.
It grants noncommercial research/academic/educational use subject to its terms.
Retain the source copyright, license and required project attribution. Any
license on separately authored reconstruction code does not relicense these
third-party components.

Project attribution: CUHK AIoT Lab, *A Large-Scale Multimodal Dataset and
Benchmarks for Human Activity Scene Understanding and Reasoning*, MobiSys 2026.
See the official repository's `CITATION.cff` and https://doi.org/10.1145/3745756.3809209.

See the [native encoder provenance](../../source/upstream/PROVENANCE_AUDIT.md)
for the original verification checkpoint.
