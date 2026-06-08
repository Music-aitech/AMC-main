# Low-Resource Video-Conditioned Music Generation via Reliability-Aware Visual-Text-Audio Fusion

This repository provides an inspectable PyTorch implementation and reproducibility
resources for **AMC**, the framework described in the paper above. The training
path can be checked end to end with synthetic tensors.

## Method Summary

AMC combines **TAM + AMF + CMTs + TSF**:

- **TAM** performs temporal alignment with blockwise temporal attention.
- **AMF** is Adaptive Multimodal Fusion, a reliability-aware fusion mechanism.
- **CMTs** are cross-modal transformer modules. The released lightweight CMT
  implementation uses stable temporal self-attention for reproducibility and
  smoke testing.
- **TSF** is temporal semantic fusion used during training to aggregate a global
  music representation.

The model accepts aligned visual sequence `v`, acoustic/music-context sequence
`a`, and text embedding sequence `t`.

## Repository Structure

```text
model.py, modules.py, train.py  AMC source code and synthetic smoke test
configs/                        training, evaluation, and ablation configurations
scripts/                        inference and evaluation entry points
metadata/                       schema, split manifests, and annotation resources
results/                        manuscript-reported summary tables
demo_examples/                  Figure 7 qualitative scene documentation
human_study/                    rating materials and manuscript MOS summary
```

## Environment Setup

Python 3.10 is recommended.

```bash
conda env create -f environment.yml
conda activate amc
```

Alternatively:

```bash
pip install -r requirements.txt
```

## Smoke Test

The smoke test creates synthetic random tensors for `v`, `a`, `t`, and a target
latent, then runs forward, loss, backward, and one AdamW optimizer step.

```bash
python train.py --smoke-test
python train.py --config configs/train_amc.yaml --smoke-test
```

## Inference

Inspect the expected inference inputs:

```bash
python scripts/infer_amc.py --help
python scripts/infer_amc.py --config configs/train_amc.yaml --smoke-test
```

## Evaluation

Summary mode audits the manuscript-reported tables:

```bash
python scripts/evaluate_fad_kl.py --summary results/table2_indomain_results.csv
python scripts/evaluate_tempo_error.py --summary results/table6_visual_perturbation.csv
python scripts/evaluate_clap_similarity.py --summary results/table7_missing_modality.csv
python scripts/evaluate_nn_similarity.py --summary results/table5_nn_similarity.csv
```

## Reproducing Manuscript Tables

Files in `results/*.csv` reproduce manuscript-reported summary tables. Open the
CSV files directly or use the summary-mode evaluation scripts.
See [results/README.md](results/README.md) for the table mapping.

## Data Availability

This repository organizes source code, configuration files, metadata resources,
evaluation scripts, manuscript-reported summary results, and qualitative scene
documentation. Corpus and media licensing information is documented in
`license_note.md` and `metadata/license_note.md`.

## Demo Examples

`demo_examples/` contains scene-specific prompt documentation corresponding to
the Figure 7 qualitative visualization.

## License And Responsible Use

The released code is under the MIT License. Generated music should be reviewed
for memorization, similarity, attribution, consent, and applicable copyright or
platform requirements. See `license_note.md`, `metadata/license_note.md`, and
`model_card.md`.
