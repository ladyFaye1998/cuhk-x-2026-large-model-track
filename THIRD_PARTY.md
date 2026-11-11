# Third-party models and source

| Component | Origin and license | Release reference |
|---|---|---|
| V-JEPA 2 video classifier | [Meta model card](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2), MIT | [Pinned revision and license](checkpoints/temporal_backbones/MODEL_NOTICES.md) |
| SigLIP image/text encoder | [Google model card](https://huggingface.co/google/siglip-base-patch16-224), Apache 2.0 | [Pinned revision and license](checkpoints/temporal_backbones/MODEL_NOTICES.md) |
| SigLIP2 image encoder | [Google model card](https://huggingface.co/google/siglip2-so400m-patch16-384), Apache 2.0 | The authorized verification bundle contains the extraction manifest for the exact image tower. |
| Compact question graph | [Original Kaggle source](https://www.kaggle.com/code/phuongncn/lb-0-77777-exact-submission-research-handoff), Apache 2.0 attribution retained in source | `code/recovered/compact_graph.py` |
| Skeleton model and teacher components | CUHK-X, Active3DPose, EfficientGCN and SkateFormer | [CUHK-X source and asset notice](components/atomic_actions/third_party/CUHK-X/NOTICE.md). The other components retain the license files alongside their source. |

The authorized verification bundle provides the model artifacts at the exact revisions used for inference, including the SigLIP2 extraction manifest and original tensor byte ranges. The inference runtime loads floating-point weights in the recorded precision.

PyTorch, Transformers, NumPy, SciPy, pandas, scikit-learn, OpenCV and Z3 are runtime dependencies. Their exact versions are recorded in `environment/runtime.json`; their upstream licenses remain applicable.
