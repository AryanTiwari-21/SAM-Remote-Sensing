# 🛰️ Segment Anything Model for Remote Sensing

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)
![SAM](https://img.shields.io/badge/SAM-Meta_AI-blueviolet)
![GroundingDINO](https://img.shields.io/badge/GroundingDINO-IDEA_Research-green)
![Colab](https://img.shields.io/badge/Run_on-Google_Colab-F9AB00?logo=googlecolab)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AryanTiwari21/SAM-Remote-Sensing)

> 🚀 **[Try the Live Demo →](https://huggingface.co/spaces/AryanTiwari21/SAM-Remote-Sensing)**

> Reproducing and extending **"The Segment Anything Model (SAM) for remote sensing applications: From zero to one-shot"** — implementing and comparing multiple segmentation strategies for building extraction on the LoveDA satellite dataset.

---

## 📌 Overview

This project explores how SAM and its variants can be applied to **remote sensing satellite imagery** for building segmentation. We evaluate four distinct approaches — ranging from zero-shot automatic segmentation to one-shot personalized segmentation — and propose a custom modification that combines detection-based prompting with morphological refinement.

**Use cases:** Urban planning, disaster response mapping, land-use analysis, infrastructure monitoring.

---

## 📊 Results

| Method | IoU |
|---|---|
| Auto Segmentation (SAM) | 0.516 |
| Multi-point Segmentation | 0.569 |
| Text-Prompt (GroundedSAM) | 0.489 |
| One-Shot (PerSAM) | **0.586** |
| Proposed (Custom Modification) | 0.522 |

> PerSAM achieves the best IoU (0.586), while Multi-point segmentation offers a strong balance between automation and accuracy.

---

## 🔧 Project Pipeline

### Phase 1 — SAM: Zero-Shot Segmentation
- **Automatic segmentation** using `segment-geospatial` (SamGeo) with no prompts
- **Point-prompt segmentation** using centroid points derived from ground truth masks

### Phase 2 — Text-Prompt Segmentation (GroundedSAM)
- **GroundingDINO** detects bounding boxes using the text query `"building"`
- **SAM** generates masks within the detected bounding boxes

### Phase 3 — One-Shot Segmentation (PerSAM)
- **Personalized SAM** uses a single reference image + mask to segment similar objects in new images
- No retraining required — purely inference-time adaptation

### Phase 4 — Custom Modification (Proposed)
- Combines GroundingDINO bounding boxes with **positive + negative point prompts**
- Applies **morphological refinement** (opening/closing operations)
- Connected component filtering to remove noise
- Bounding box centers used as automatic positive point prompts for SAM

---

## 🗂️ Repository Structure

```
├── SAMRunner.ipynb        # Main notebook with all 4 phases
├── README.md
└── outputs/
    ├── samgeo_mask.png           # Phase 1: Auto segmentation output
    ├── multipoint_mask.png       # Phase 1: Multi-point output
    ├── final_multipoint_mask.png # Phase 1: Refined multi-point output
    ├── pred_mask.png             # Phase 2: Text-prompt output
    ├── persam_output/            # Phase 3: PerSAM outputs
    │   └── building/
    └── phase4_final_mask.png     # Phase 4: Custom modification output
```

---

## ⚙️ Setup & How to Run

### 1. Clone required repositories in Google Colab

```bash
git clone https://github.com/opengeos/segment-geospatial.git
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
git clone https://github.com/ZrrSkywalker/Personalize-SAM.git
```

### 2. Open the notebook

Open `SAMRunner.ipynb` in [Google Colab](https://colab.research.google.com/) and enable **GPU runtime** (Runtime → Change runtime type → T4 GPU).

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Download the dataset

Download the **LoveDA** dataset from the [official source](https://github.com/Junjue-Wang/LoveDA) and place it in your Google Drive. Update the dataset path in the notebook accordingly.

### 5. Run all cells sequentially

Pretrained model weights (SAM ViT-H, GroundingDINO) will be downloaded automatically.

---

## 📦 Dataset

| Property | Details |
|---|---|
| Name | LoveDA (Land-cover Dataset for Domain Adaptive Semantic Segmentation) |
| Task | Building segmentation |
| Scene | Urban satellite imagery |
| Included | ❌ Not included (download separately) |

---

## 🧰 Tech Stack

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) — Meta AI
- [segment-geospatial (SamGeo)](https://github.com/opengeos/segment-geospatial) — Geospatial wrapper for SAM
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) — GroundingDINO + SAM
- [Personalize-SAM (PerSAM)](https://github.com/ZrrSkywalker/Personalize-SAM) — One-shot SAM adaptation
- PyTorch, OpenCV, NumPy, Matplotlib

---

## 👨‍💻 Authors

| Name | Roll No. |
|---|---|
| Aryan Tiwari | 2446026 |
| Anant Pandey | 2446029 |
| Harsh Chhapre | 2446030 |

---

## 📄 Base Paper

This project reproduces and extends the following paper:

> **Osco, L.P., Wu, Q., de Lemos, E.L., Gonçalves, W.N., Ramos, A.P.M., Li, J., & Marcato Junior, J. (2023).**
> *The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot.*
> International Journal of Applied Earth Observation and Geoinformation, 124, 103540.
> [https://doi.org/10.1016/j.jag.2023.103540](https://doi.org/10.1016/j.jag.2023.103540)

```bibtex
@article{osco2023segment,
  title={The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot},
  author={Osco, Lucas Prado and Wu, Qiusheng and de Lemos, Eduardo Lopes and
          Gonçalves, Wesley Nunes and Ramos, Ana Paula Marques and
          Li, Jonathan and Marcato Junior, José},
  journal={International Journal of Applied Earth Observation and Geoinformation},
  volume={124},
  pages={103540},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.jag.2023.103540}
}
```

## 📚 Additional References

- Kirillov, A. et al. (2023). [*Segment Anything*](https://arxiv.org/abs/2304.02643) — Meta AI
- Liu, S. et al. (2023). [*Grounding DINO: Marrying DINO with Grounded Pre-Training*](https://arxiv.org/abs/2303.05499)
- Zhang, R. et al. (2023). [*Personalize Segment Anything Model with One Shot*](https://arxiv.org/abs/2305.03048)
- Wang, J. et al. (2022). [*LoveDA: A Remote Sensing Land-Cover Dataset*](https://arxiv.org/abs/2110.08733)
