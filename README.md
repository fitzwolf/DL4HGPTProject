# TRACE Reproduction and Simulation Project

## 📄 Citation

This project is based on the TRACE methodology described in the original paper:

Weatherhead, Addison, et al. *"Learning Unsupervised Representations for ICU Timeseries."* Proceedings of the Conference on Health, Inference, and Learning. Vol. 174. PMLR, 2022. [Link to Paper](https://proceedings.mlr.press/v174/weatherhead22a/weatherhead22a.pdf)

## 📁 Repository Structure

### `/code/` — Core Python Modules

This directory contains all modularized components used to run TRACE-style representation learning on synthetic ICU data.

#### `simulate_patient.py`

- Contains functions to generate realistic synthetic ICU patients.
- Simulates different clinical states like healthy, shock, and respiratory failure.
- Supports generating multiple runs and saving to disk in `run_0`, `run_1`, ... format.
- Also includes a longevity scoring heuristic.

#### `dataset.py`

- Implements two PyTorch-compatible `Dataset` classes:
  - `RichSyntheticICUDataset`: loads `.npy` ICU simulation runs.
  - `LongevityDataset`: for longevity prediction task.
- Used in data loaders for training and evaluation.

#### `model.py`

- Defines the TRACE-style masked convolutional encoder model (`TRACEEncoderMasked`).
- Includes support functions for masked modeling:
  - `generate_random_mask`
  - `masked_mse_loss`
- Also defines the transfer classifier architecture (`TransferClassifier`).

#### `train.py`

- Functions for model training and evaluation:
  - `train_trace_masked_model`: self-supervised training of TRACE.
  - `extract_embeddings_and_labels`: get embeddings from encoder.
  - `train_transfer_classifier`: train classifier on frozen encoder.
- Includes utilities for:
  - t-SNE visualization
  - Classifier benchmarking (logistic regression, random forest, k-NN)
  - Flattening signals for raw baseline comparison

## 📓 Jupyter Notebooks

### `DL4HRecreationSimulated.ipynb`

- End-to-end recreation of the TRACE pipeline using synthetic ICU data.
- Trains a TRACE encoder via masked modeling on `RichSyntheticICUDataset`.
- Evaluates representation quality using:
  - t-SNE plots
  - Classification benchmarks
- Includes experiments on raw signal vs embedding performance.

### `DL4HFinalProject.ipynb`

- Full experimental workflow demonstrating transfer learning.
- Builds on the trained TRACE encoder and freezes it.
- Adds classifier head to predict:
  - Health condition (0: healthy, 1: shock, 2: respiratory failure)
  - Longevity prediction (binary)
- Visualizes t-SNE embedding separation for longevity.
- Includes noise robustness and feature dropout experiments.

---

## ▶️ Getting Started

```bash
# Install required packages
pip install -r requirements.txt

Running to code might fail, the notebooks should work inside a colab environment.