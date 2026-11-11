# Multimodal Motion Reasoning with Temporal and Structured Fusion

CUHK-X Challenge 2026, Large Model Track (L03)  
Knight of Favonius · Danielle Lesin

The solution combines multimodal multiple-instance inference, a temporal video ensemble, a skeleton-based atomic-action classifier, recording-family transfer, and constrained decoding for question answering. A final question-aware classifier produces one prediction per official question.

The AIoT Lab, Department of Information Engineering, The Chinese University of Hong Kong, created CUHK-X. Cite [Jiang et al., MobiSys 2026](https://doi.org/10.1145/3745756.3809209) when using the dataset.

## Reproduce inference

1. Obtain access to the [official CUHK-X challenge inputs](https://openaiotlab.github.io/CUHK-X-Challenge/dataset.html). Under one input directory, make available the official question table (`test_qa.csv`), the Large Model test video archive (`large_model_track_test.zip`), the Large Model sensor archive (`large_sensors.zip`), and the Small Model test clip archive (`small_model_track_test.zip`). The input resolver also supports single-file archives with the `.zip.001` suffix.
2. Obtain authorized access to the verified model assets and training data required for inference. The organizers have the original verification package and Kaggle assets. See [THIRD_PARTY.md](THIRD_PARTY.md) for source, license, and input details. The original Kaggle environment used one Tesla T4.
3. Unpack the authorized verification package and run these commands from its `CUHKX_large` directory on a CUDA-capable Linux machine with Python 3.12.

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r environment/requirements-runtime.txt
   python code/run.py --package . --data-dir /absolute/path/to/official-data --output /absolute/path/to/new-run
   ```

The command writes `submission.csv` and `execution.json` to the new output directory.

For the [Kaggle notebook](cuhk-x-large-polished-motion-classifier.ipynb), mount the official inputs and the inference, recording-model, and option-head assets in a fresh GPU session. On a CUDA-capable Linux machine, make the same asset directories available beneath `CUHKX_INPUT`. The notebook locates them by their manifests. To execute the notebook locally from this repository, activate the same Python environment and run these commands.

   ```bash
   python -m pip install jupyter
   export CUHKX_PYTHON="$(command -v python)"
   export CUHKX_INPUT="/absolute/path/to/inputs"
   export CUHKX_OUTPUT="/absolute/path/to/new-run"
   mkdir -p "$CUHKX_OUTPUT"
   jupyter nbconvert --to notebook --execute \
     --ExecutePreprocessor.timeout=-1 \
     --output-dir "$CUHKX_OUTPUT" --output executed-inference.ipynb \
     cuhk-x-large-polished-motion-classifier.ipynb
   ```

The notebook checks the input identities, generates the raw model features and predictions, and writes `submission.csv` in question order. It also saves `inference_manifest.json`, `runtime.json`, and the inference log under the output directory.

## Source layout

- `code/` contains the Large Model inference and model-fitting code, including the final question-aware head.
- `recording_model/` contains the recording-transfer orchestration and model logic.
- `components/atomic_actions/` contains the skeleton-based atomic-action branch and its training code.
- `training/` contains temporal fitting, recording-model fitting, and option-head training source.
- `environment/` records the inference dependencies and runtime versions.

## License

Original solution code is available under the [Apache License 2.0](LICENSE). Third-party components retain their accompanying licenses and notices. Obtain CUHK-X observations from the organizers under the dataset terms. Competition data, trained weights, derived input bundles, and generated predictions are outside this source release.
