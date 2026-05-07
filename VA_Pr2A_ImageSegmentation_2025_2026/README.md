# Satellite Image Segmentation — xBD Building Damage Assessment

**Visión Artificial · UC3M · 2025–2026**  
Juan Macías Romero · Manuel Andrés Trujillo · Sofía Carmona Fuentes

Semantic segmentation of post-disaster satellite imagery using the xBD dataset. The project progresses from a custom Encoder-Decoder baseline to fine-tuned DeepLabV3 (ResNet-101), exploring binary building detection and multiclass damage classification (no-damage / minor / major / destroyed).

---

## Dataset

**xBD (xView2 Building Damage Assessment)**  
Satellite imagery of natural disaster zones with per-building damage labels.

The UC3M subset (`xBD_UC3M/`) is organized as follows:

```
xBD_UC3M/
├── train/   # 256 images
├── val/     # 45 images
└── test/    # 63 images
```

Each split contains disaster-named subdirectories (e.g. `palu-tsunami/`, `mexico-earthquake/`) with `images/` and `labels/` folders. Labels are JSON files with per-building polygon annotations and damage subtypes.

The dataset is not included in this repository. Place the `xBD_UC3M/` folder at the project root before running.

---

## How to Run

**Requirements**: Conda environment `cv-seg`.

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate cv-seg

# Launch Jupyter
jupyter lab
```

Open the notebooks in order:

| Notebook | Content |
|---|---|
| `APAI_Pr2A_ImageSegmentation_2025_2026.ipynb` | Parts 1–3: dataset exploration, binary segmentation (EXP-01 to EXP-04) |
| `APAI_Pr2A_ImageSegmentation_2025_202-part4.ipynb` | Part 4: multiclass damage classification (EXP-05, EXP-06) |

Results, training logs, and visualizations are saved to `results/parte-4/<exp_name>/`.

---

## Our experiments' results

| | Model | Task | Jaccard |
|---|---|---|---|
| EXP-01 | Encoder-Decoder (custom) | Binary | 0.346 |
| EXP-02 | U-Net + Weighted CE | Binary | 0.427 |
| EXP-03 | U-Net + Focal Loss + Augmentation | Binary | 0.441 |
| EXP-04 | DeepLabV3 R101 fine-tuning | Binary | 0.494 |
| EXP-05 | DeepLabV3 R101 multiclass | Multiclass (4 fg) | 0.099 |
| EXP-06 | DeepLabV3 R101 + ASPP[6,12,18] + aux loss | Multiclass (4 fg) | 0.092 |

See `results/parte-4/experiments.md` for full analysis.
