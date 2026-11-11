# Atomic-action evidence for Large inference

This classifier is reused from the project's Small-track model. It labels raw
Skeleton clips; the Large clip matcher uses their infrared/depth frames to locate
the corresponding actions and times inside Large recordings. The Large coordinator
calls `code/raw_clips.py` with this component directory.

The original verification package supplies `checkpoints/model.pth` with the
encoder, normalization, and fitted affine head. `source/raw_inputs.py` and
`source/student_training.py` provide preprocessing and prediction with
the [external CUHK-X DSTformer modules](third_party/CUHK-X/NOTICE.md).

`source/upstream/` contains the teacher/student fitting source. Training
requires the external native initialization, the organizer's raw observations,
and the official sample template. From the verification package, this command
creates the fitted stages in a new directory and produces
`result/downstream/skeleton_model.pt` plus its training records.

```bash
python components/atomic_actions/source/upstream/end_to_end.py \
  --run-root /work/new_action_training \
  --native-input components/atomic_actions/assets/native/dstformer_best_epoch.bin \
  --template /data/official_small_sample_submission.csv
```

The teacher training runtime is documented in `source/upstream/ENVIRONMENT.md`.
Methods, initialization provenance and source licenses remain with this component.
