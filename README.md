# MammoCBM: A Multiparameter Concept-Based Interpretable Model for Early Breast Cancer Diagnosis and Structured Reporting

This is the official repository for the paper **"Multiparameter concept-based interpretable model for early breast cancer diagnosis and structured reporting: a multi-center, multi-reader, radiologist-in-the-loop study"** (BMC Medicine, 2026).

📄 **Paper:** https://link.springer.com/article/10.1186/s12916-026-04889-7

## Overview

MammoCBM is an intrinsically interpretable **Concept Bottleneck Model (CBM)** for differentiating early-stage breast cancer (BC) from benign lesions on multiparametric breast MRI. Unlike conventional black-box deep learning models, MammoCBM first predicts a set of clinically meaningful, **BI-RADS-compliant concepts** (e.g., *irregular shape*, *spiculated margin*, *heterogeneous enhancement*) from the images, then performs the final benign/malignant classification as a **linear combination of these concepts**. This mirrors how radiologists reason and makes each prediction traceable to human-interpretable evidence.

An embedded large language model (GPT-5 / GPT-4o) transforms the model's concept activations and malignancy probability into a **structured BI-RADS report** containing standardized lesion descriptions, malignancy probability, BI-RADS category, and management recommendations.

## Method

The pipeline is a two-stage, multimodal, interpretable architecture:

```
                 ADC ─┐
                 CE  ─┤   per-modality           attention        concept          linear
   Multiparam.   DWI ─┼─► 3D EfficientNet ──►    pooling    ──►   predictors  ──►  classifier ──► P(malignant)
   Breast MRI    T2WI─┤   encoders               fusion          (47 BI-RADS       (interpretable
                 TIC ─┘   (TIC via MLP)                           concepts)          weights)
                                                                       │
                                                                       ▼
                                                                  GPT-5 / GPT-4o
                                                                       │
                                                                       ▼
                                                          Structured BI-RADS report
```

1. **Sequence-specific encoders** — A 3D EfficientNet backbone (MONAI) encodes each imaging sequence (T2WI, DWI, ADC, CE). Kinetic information from the **time–intensity curve (TIC)** is extracted with GPT-4o, converted to one-hot labels, and encoded by an MLP.
2. **Attention-pooling fusion** — Features from all five modalities are merged with an attention-pooling module into a joint representation.
3. **Concept layer** — One linear (binary) classifier is trained per concept to project image representations into the concept space, producing an **alignment score** for each of the **47 standardized BI-RADS concepts**.
4. **Interpretable classifier** — Concept alignment scores are fed to a sparse linear classifier (with L1 regularization) that estimates malignancy probability. The classifier weights directly quantify each concept's contribution.
5. **Report generation** — GPT-5 converts the concept activations and predicted probability into a structured, BI-RADS-compliant clinical report.

The 47-item concept bank was built by prompting GPT-4o to extract lesion descriptors from free-text reports, standardizing them to the BI-RADS lexicon, and validating them with radiologists (see `chatGPT/BIRADS.json`).

## Repository Structure

```
MammoCBM/
├── train_efficientNet.py        # Stage 1: train the black-box multimodal EfficientNet encoder
├── train_CBM.py                 # Stage 2: train the Concept Bottleneck Model on frozen features
├── train_OCCCBM.py              # Occurrence-CBM variant (spatial concept localization, 3D occ maps)
├── train_efficient_concept.py   # Concept-supervised EfficientNet + per-concept GradCAM
├── infer_CBM.py                 # Inference + structured BI-RADS report generation
├── reportProcess.py             # LLM pipeline: free-text → structure → report; TIC extraction
├── multiRun.py                  # Multi-GPU / multi-fold grid-search job launcher
├── analysis.py                  # t-SNE visualization of cached feature embeddings
├── debug.py                     # Developer sanity checks
├── bash_train_cbm.sh            # Example CBM training commands
├── bash_train_effi.sh           # Example EfficientNet training commands
│
├── chatGPT/                     # LLM interfaces & BI-RADS prompt templates
│   ├── chatgpt.py               # OpenAI client wrapper
│   ├── gemini.py                # Gemini wrapper
│   ├── prompt.py                # Prompt builder (free-text→structure, TIC, report)
│   ├── BIRADS.json / BIRADS.txt # 47-item BI-RADS concept schema
│   ├── TIC_Prompt.txt           # Prompt for TIC-curve point extraction
│   └── Report_Prompt_REVISED.md # Report-generation prompt (BI-RADS table)
│
├── config/                      # Hierarchical mmengine configs (with _base_ inheritance)
│   ├── config.py, param.py      # Config class + argparse; global params
│   ├── InternalData/            # {BlackBox,CBM}/*.py per-model configs
│   ├── InternalDataBreast/      # breast-cropped variant
│   ├── InternalDataAntsBreast/  # ANTs-registered breast-segmented variant
│   └── Prospective/             # prospective-cohort configs
│
├── dataset/                     # Data loading & preprocessing (data itself is gitignored)
│   ├── dataloader.py            # DataModule, Dataset, FeatureDataset, ConceptDataset
│   ├── data_split.py            # DataSplit: stratified k-fold CSV generation
│   ├── transforms.py            # MONAI-based 2D/3D transforms
│   └── dicom.py, preprocess.py, medsam3d.py, utils.py
│
├── models/
│   ├── __init__.py              # get_model_from_config() factory
│   ├── module.py                # MultiModalModule (PyTorch Lightning base)
│   ├── backbone/                # MultiModels + networks (efficientnet, vit, swin, resnet, ...)
│   ├── cbm/                     # baseCBM, CBM, linearCBM, occCBM
│   ├── cbank/                   # ConceptsLearner / ConceptPredictor
│   └── clips/
│
├── utils/                       # trainHelper, logger, metrics, checkpoint, grad_cam, tsne, ...
│
└── mamclip/                     # Embedded Mammo-CLIP (MICCAI 2024) foundation-model codebase
                                 # (third-party dependency; see mamclip/README.md)
```

## Installation

Requires **Python 3.11** and a CUDA-capable GPU.

```bash
git clone https://github.com/ly1998117/MMCBM.git MammoCBM
cd MammoCBM

# create environment
conda create -n mammocbm python=3.11 -y
conda activate mammocbm

# install dependencies
pip install -r requirements.txt
```

### Dependencies

The project relies on the following main packages (no lock file is shipped; install the latest compatible versions):

**Deep learning / core**
- `torch`, `torchvision`, `torchmetrics`
- `lightning` (PyTorch Lightning 2.x)
- `monai` (3D transforms, networks, EfficientNet)
- `timm`, `transformers`
- `mmengine` (configuration system)

**Medical imaging / data**
- `nibabel`, `SimpleITK`, `pydicom`, `dicomsdl`, `antspyx`
- `opencv-python`, `Pillow`, `scipy`, `scikit-learn`, `numpy`, `pandas`

**LLM / reporting**
- `openai`, `google-generativeai`

**Utilities**
- `mpire`, `redis`, `tqdm`, `matplotlib`, `hydra-core`, `omegaconf`, `fire`, `albumentations`, `nltk`

> The `mamclip/` subdirectory is a self-contained copy of the third-party **Mammo-CLIP** codebase and has its own environment/instructions in `mamclip/README.md`.

## Data Preparation

Data is expected under `dataset/{DATASET}/` (e.g. `dataset/InternalData/`). The directory is gitignored; only the `.py` loaders are tracked.

- **Modalities:** `ADC`, `CE`, `DWI`, `T2WI` (3D NIfTI `.nii.gz`, resized to 128³) plus `TIC` (kinetic-curve feature vector). `MM` denotes all modalities combined.
- **Lesion annotation:** a 3D rectangular bounding box (from `BOX.nii.gz`) enclosing each lesion, propagated across the co-registered sequences.
- **CSVs** under `dataset/{DATASET}/CSV/`:
  - `data_split/datalist.csv` — master list with columns `name`, `pid`, `pathology` (`Benign`/`BC`), `modality`, `path` (dict mapping modality → file path), and for concept models `concept_key`, `concept_label` (and optional `bbox`).
  - `data_split/{train,test,k_fold}.csv` — auto-generated by `DataSplit`.
  - `report/{report,concepts,tic,TIC-GT}.csv` — for the LLM reporting pipeline.
- **Labels:** `pathology_labels = {'Benign': 0, 'BC': 1}`.
- **Concepts:** 47 BI-RADS attributes defined in `chatGPT/BIRADS.json` (the `BI-RADS_category` group is excluded during concept learning).

## Usage

Training is orchestrated through `multiRun.py`, which reads an mmengine `.py` config and can launch runs across GPUs and folds. Individual training scripts may also be invoked directly.

### Stage 1 — Train the black-box multimodal encoder

```bash
python multiRun.py \
  --script train_efficientNet.py \
  --options config=config/InternalData/BlackBox/efficient.py \
  --func search_fold \
  --device 4,5,6,7 --n_jobs 1
```

### Stage 2 — Train the Concept Bottleneck Model

Use the encoder checkpoint from Stage 1 as `encode_dir`:

```bash
python multiRun.py \
  --script train_CBM.py \
  --options config=config/InternalData/CBM/mmcbm.py \
  --options encode_dir='result/InternalData_Zeros/efficientnet-b0_spool_iterTrain_LR0.0001_fullshot' \
  --func search_fold \
  --device 3,4,5,6,7 --n_jobs 1
```

See `bash_train_cbm.sh` and `bash_train_effi.sh` for additional ready-to-run examples (including the Occurrence-CBM and ANTs-breast variants).

### Occurrence-CBM (spatial concept localization)

```bash
python multiRun.py \
  --script train_CBM.py \
  --config config/InternalData/CBM/occcbm.py \
  --func search_param_loss \
  --device 4,5,6,7 --n_jobs 1
```

### Inference & structured report generation

```bash
python infer_CBM.py
```

This runs the trained CBM on a data module and generates structured BI-RADS reports via `ReportProcess` (`reportProcess.py`), which calls the configured LLM.

> **Security note:** configure LLM credentials via environment variables / a local config rather than hard-coding them. Do **not** commit API keys.

### Useful CLI options

Parsed by `config/config.py` (`Config.config()`); can also be passed via `--options key=value`:

| Option | Description |
|---|---|
| `--config` | Path to the mmengine config file |
| `--modality` | Modality to use (default `MM`) |
| `--device` | GPU id(s), comma-separated |
| `-k` / `--k` | Cross-validation fold index |
| `--test` | Run in test mode |
| `--resume` | Resume from checkpoint |
| `--infer` | Run inference |
| `--cache_data` | Pre-encode & cache features |
| `--no_data_aug` | Disable data augmentation |
| `--plot`, `--plot_curve` | Enable plotting / TIC-curve plots |
| `--occ_act` | Occurrence-map activation (`abs`/`sigmoid`/`softmax`) |
| `--options` | mmengine dotted-key config overrides |

### Configuration highlights

- **Black-box** (`config/*/BlackBox/*.py`): `model_name='efficientnet-b0'`, `fusion_method='spool'`, `epochs=300`, `batch_size=4`, `lr=1e-4`, `img_size=128`, `spatial_dims=3`, `num_class=2`.
- **CBM** (`config/*/CBM/*.py`): `lambda_l1=0.001` (concept sparsity), `pre_encode_to_feature=True`, `epochs=1000`, `batch_size=32`, `lr=5e-5`. Model variants: `cbm`, `mmcbm`, `mm2cbm`, `occcbm`.

## Model Zoo / Outputs

Training writes to `result/…` directories whose names encode dataset, weight-init, L1, normalization, LR, shot count, fusion method, and activation. Test outputs (metrics, predictions, concept scores) are dumped as CSV/`.pth`. Occurrence-CBM `infer()` additionally exports 3D NIfTI occurrence and feature maps for spatial concept visualization.

## Metrics

Reported per modality via `torchmetrics`: **Accuracy, AUROC, Recall, Precision, F1** (binary for classification, multilabel for concepts).

## Acknowledgements

- The `mamclip/` directory contains a copy of **Mammo-CLIP** (Ghosh et al., MICCAI 2024) used as a foundation-model dependency. See `mamclip/README.md` for its license, setup, and citation.
- The CBM design builds on the Hybrid Concept Bottleneck Model line of work (Liu et al., CVPR 2025).

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{qu2026mammocbm,
  title   = {Multiparameter concept-based interpretable model for early breast cancer diagnosis and structured reporting: a multi-center, multi-reader, radiologist-in-the-loop study},
  author  = {Qu, Jiao and Liu, Yang and Zhang, Mengmei and Xie, Huan and Yang, Juan and Xu, Mengyuan and Yang, Ling and Qian, Wenlei and Li, Yan and Jiang, Jing and Zhou, Lin and Zeng, Jiaxin and Tu, Danyang and Zhang, Jiajin and Sun, Jiachen and Huang, Juan and Jing, Jing and Shi, Hubing and Gu, Shi and Lui, Su},
  journal = {BMC Medicine},
  volume  = {24},
  number  = {345},
  year    = {2026},
  doi     = {10.1186/s12916-026-04889-7},
  url     = {https://link.springer.com/article/10.1186/s12916-026-04889-7}
}
```
