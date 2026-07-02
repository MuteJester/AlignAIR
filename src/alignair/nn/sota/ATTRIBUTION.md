# Attribution

The modules in `alignair/nn/sota/` are clean 1‑D PyTorch re-implementations of ideas from the
following permissively-licensed projects (cloned to `.private/reference/` for study). They are
adapted to nucleotide sequences and our task; no framework code is vendored. Where an algorithm
is taken closely, the source is named in the module docstring.

| Module | Idea | Source | License |
|--------|------|--------|---------|
| `matching.py` | late-interaction MaxSim; symmetric InfoNCE + learnable `logit_scale` | ColBERT; **GLIP** region-word; **open_clip** | MIT |
| `fusion.py` | bidirectional vision-language cross-attention (`BiAttentionBlock`) | **GLIP** (`maskrcnn_benchmark/utils/fuse_helper.py`) | MIT |
| `query_decoder.py` | object-query transformer decoder | **DETR** (`models/transformer.py`) | Apache-2.0 |
| `span_head.py` | decoupled cls/reg/objectness head | **YOLOX** (`yolox/models/yolo_head.py`) | Apache-2.0 |
| `loss.py` | set-criterion box loss (L1 + generalized-IoU), objectness BCE | **DETR** (`models/detr.py` `SetCriterion`); **YOLOX** (`yolo_head.py`) | Apache-2.0 |
| `detector.py` | assembly (encoder → fusion → typed queries → decoupled heads) | our composition of the above (YOLO-World / GLIP shape) | — |

- GLIP — Microsoft, MIT — https://github.com/microsoft/GLIP
- open_clip — ML Foundations, MIT — https://github.com/mlfoundations/open_clip
- DETR — Facebook Research, Apache-2.0 — https://github.com/facebookresearch/detr
- YOLOX — Megvii, Apache-2.0 — https://github.com/Megvii-BaseDetection/YOLOX

We deliberately do **not** use Ultralytics YOLO (AGPL-3.0) or Nucleotide-Transformer weights
(CC-BY-NC).
