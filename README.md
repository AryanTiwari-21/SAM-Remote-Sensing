# 🛰️ Segment Anything Model for Remote Sensing

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?logo=pytorch)](https://pytorch.org/)
[![SAM](https://img.shields.io/badge/SAM-Meta_AI-blueviolet)](https://github.com/facebookresearch/segment-anything)
[![GroundingDINO](https://img.shields.io/badge/GroundingDINO-IDEA_Research-green)](https://github.com/IDEA-Research/Grounded-Segment-Anything)
[![Run on Google Colab](https://img.shields.io/badge/Run_on-Google_Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AryanTiwari21/SAM-Remote-Sensing)

> 🚀 **[Try the Live Demo →](https://huggingface.co/spaces/AryanTiwari21/SAM-Remote-Sensing)**

> Reproducing and extending **"The Segment Anything Model (SAM) for remote sensing applications: From zero to one-shot"** — implementing and comparing multiple segmentation strategies for building extraction on the LoveDA satellite dataset.

---

## 📌 Overview

This project explores how SAM and its variants can be applied to **remote sensing satellite imagery** for building segmentation. We evaluate four distinct approaches — ranging from zero-shot automatic segmentation to one-shot personalized segmentation — and propose a custom modification that combines detection-based prompting with morphological refinement.

**Use cases:** Urban planning, disaster response mapping, land-use analysis, infrastructure monitoring.

---

## 📊 Results

| Method                         | IoU       |
| ------------------------------ | --------- |
| Auto Segmentation (SAM)        | 0.516     |
| Multi-point Segmentation       | 0.569     |
| Text-Prompt (GroundedSAM)      | 0.489     |
| One-Shot (PerSAM)              | **0.586** |
| Proposed (Custom Modification) | 0.522     |

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
├── SAMRunner.ipynb              # Main research notebook (all 4 phases)
├── GradioApp.py                 # Gradio app source (local use)
├── GradioRunner.ipynb           # Notebook to launch the Gradio app via Colab
├── deployment/                  # Hugging Face Spaces deployment files
│   ├── app.py                   # Main app entry point for HF Spaces
│   ├── requirements.txt         # Python dependencies for HF Spaces
│   └── packages.txt             # System-level dependencies (apt)
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

## 🚀 Deployment

The Gradio-based interactive demo is deployed on **Hugging Face Spaces** and can also be run locally.

### 🌐 Live Demo

👉 **[https://huggingface.co/spaces/AryanTiwari21/SAM-Remote-Sensing](https://huggingface.co/spaces/AryanTiwari21/SAM-Remote-Sensing)**

Upload any satellite/aerial image and choose a segmentation mode (Auto, Multi-point, Text-Prompt, or One-Shot) to get predictions in real time — no setup required.

---

### 🖥️ Run the Gradio App Locally

#### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU fallback is supported but slow)

#### Step 1 — Clone this repository

```bash
git clone https://github.com/AryanTiwari-21/SAM-Remote-Sensing.git
cd SAM-Remote-Sensing
```

#### Step 2 — Install system dependencies

```bash
sudo apt-get install -y libgl1 libglib2.0-0 wget git
```

#### Step 3 — Install Python dependencies

```bash
pip install torch>=2.0.0 torchvision
pip install opencv-python-headless Pillow numpy
pip install transformers>=4.36.0 tokenizers>=0.15.0
pip install rasterio addict yapf timm supervision
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/opengeos/segment-geospatial.git
pip install gradio
```

Or install all at once using the deployment requirements file:

```bash
pip install -r deployment/requirements.txt
pip install gradio
```

#### Step 4 — Launch the app

```bash
python GradioApp.py
```

Then open your browser at `http://localhost:7860`.

---

### 📓 Run via Colab (GradioRunner.ipynb)

If you prefer not to set up a local environment, you can launch the Gradio app directly from Google Colab using the provided notebook:

1. Open `GradioRunner.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Enable **GPU runtime** (Runtime → Change runtime type → T4 GPU)
3. Run all cells — a public share link (valid for 72 hours) will be generated automatically via `share=True`

---

### ☁️ Deploy Your Own Hugging Face Space

To host your own version of this app on Hugging Face Spaces:

#### Step 1 — Create a new Space

Go to [huggingface.co/new-space](https://huggingface.co/new-space), set the SDK to **Gradio**, and choose a Space name.

#### Step 2 — Push the deployment files

```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cd <your-space-name>

# Copy deployment files
cp /path/to/SAM-Remote-Sensing/deployment/app.py .
cp /path/to/SAM-Remote-Sensing/deployment/requirements.txt .
cp /path/to/SAM-Remote-Sensing/deployment/packages.txt .
```

#### Step 3 — Commit and push

```bash
git add .
git commit -m "Initial deployment"
git push
```

The Space will build automatically. Initial builds may take 5–10 minutes due to model weight downloads.

#### Deployment File Reference

| File               | Purpose                                              |
| ------------------ | ---------------------------------------------------- |
| `app.py`           | Main Gradio application (entry point for HF Spaces)  |
| `requirements.txt` | Python packages installed during Space build         |
| `packages.txt`     | System-level apt packages (`libgl1`, `git`, etc.)    |

> **Note:** The Space runs on a CPU instance by default on Hugging Face's free tier. Inference will be slower than a GPU environment. Upgrade to a GPU-backed Space for faster results.

---

## 📦 Dataset

| Property | Details                                                               |
| -------- | --------------------------------------------------------------------- |
| Name     | LoveDA (Land-cover Dataset for Domain Adaptive Semantic Segmentation) |
| Task     | Building segmentation                                                 |
| Scene    | Urban satellite imagery                                               |
| Included | ❌ Not included (download separately)                                  |

---

## 🧰 Tech Stack

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) — Meta AI
- [segment-geospatial (SamGeo)](https://github.com/opengeos/segment-geospatial) — Geospatial wrapper for SAM
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) — GroundingDINO + SAM
- [Personalize-SAM (PerSAM)](https://github.com/ZrrSkywalker/Personalize-SAM) — One-shot SAM adaptation
- [Gradio](https://gradio.app/) — Interactive web demo framework
- PyTorch, OpenCV, NumPy, Matplotlib

---

## 👨‍💻 Authors

| Name          | Roll No. |
| ------------- | -------- |
| Aryan Tiwari  | 2446026  |
| Anant Pandey  | 2446029  |
| Harsh Chhapre | 2446030  |

---

## 📄 Base Paper

This project reproduces and extends the following paper:

> **Osco, L.P., Wu, Q., de Lemos, E.L., Gonçalves, W.N., Ramos, A.P.M., Li, J., & Marcato Junior, J. (2023).** *The Segment Anything Model (SAM) for remote sensing applications: From zero to one shot.* International Journal of Applied Earth Observation and Geoinformation, 124, 103540. <https://doi.org/10.1016/j.jag.2023.103540>

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
