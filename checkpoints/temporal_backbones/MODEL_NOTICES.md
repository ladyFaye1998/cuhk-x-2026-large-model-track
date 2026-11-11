# Temporal backbone companion

These are unchanged public model files, downloaded on 12 September 2026. All eleven files in `acquisition_report.json` match the retained preparation receipt. Both model-weight digests also match their publishers' LFS metadata.

V-JEPA 2 is by Meta Platforms, Inc. and affiliates. The pinned model is [facebook/vjepa2-vitl-fpc16-256-ssv2](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2/tree/4aa02df83918538fc21cfaf576382fa20e489a80), revision `4aa02df83918538fc21cfaf576382fa20e489a80`. The publisher labels it MIT. `vjepa2/LICENSE` preserves the [upstream copyright and permission notice](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE), retrieved on 12 September 2026. Cite the V-JEPA 2 paper by Assran et al. (2025).

SigLIP is by Google. The pinned model is [google/siglip-base-patch16-224](https://huggingface.co/google/siglip-base-patch16-224/tree/7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed), revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`. Its publisher labels it Apache-2.0. `siglip/LICENSE` is the [upstream license text](https://github.com/google-research/big_vision/blob/main/LICENSE), retrieved on 12 September 2026. Cite Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (2023).

No weights were quantized, converted, fine-tuned or otherwise modified for this delivery. Metadata and this notice were added. This companion does not contain CUHK-X raw recordings or labels, the six CUHK-X temporal task heads, or a full competition answer generator. Model delivery is not an organizer ruling or a certificate of complete pipeline compliance.

Extract this volume to obtain `temporal_backbones/vjepa2` and `temporal_backbones/siglip`. The CPU probe and public acquisition script use that directory beside their source. Each file is listed in `TEMPORAL_DELIVERY_MANIFEST.json`; the archive hash and independently extracted member checks are recorded outside the archive.
