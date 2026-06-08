# AMC Model Card

## Model

- **Name:** AMC
- **Task:** low-resource video-conditioned music generation
- **Inputs:** visual sequence, text prompt/text embedding, and acoustic/music context
- **Output:** fused AMC latent representation for waveform generation

## Training Data

AMC was studied with a 100-hour licensed-data-constrained corpus.

## Released Materials

Source code, configuration files, metadata schema, split manifests, evaluation
interfaces, manuscript-reported summary tables, human-study materials, and
qualitative scene documentation are organized in this repository.

## Limitations

Performance may vary by genre, culture, prompt specificity, visual quality, and
the reliability of each input modality.

## Intended Use

Research inspection, implementation auditing, controlled experimentation, and
extension of reliability-aware multimodal music generation.

## Non-Intended Use

Do not use AMC to imitate living artists without permission, falsely attribute
generated music, bypass media licenses, or claim that outputs are universally
copyright-free. Do not deploy it in high-stakes settings without additional
validation and governance.
