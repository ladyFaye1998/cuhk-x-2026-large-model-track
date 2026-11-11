# Atomic-action classifier training environment

The teacher/student training coordinator is `end_to_end.py`. It provisions the
stage-specific environments specified in `runtime_setup.py`, preserving an
environment receipt in its output directory. The recorded training run used the
Kaggle Linux image
`gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
and one Tesla T4. Fitting took approximately 5.7 GPU hours, excluding CPU stages,
downloads and environment setup. `CUHKX_RUNTIME_TEMP` selects temporary storage.

Large inference uses the enclosing package's `environment/requirements-inference.txt`.
The classifier itself runs on CPU with deterministic preprocessing and fixed seeds.
Retraining across different hardware or library builds can change fitted parameters.
