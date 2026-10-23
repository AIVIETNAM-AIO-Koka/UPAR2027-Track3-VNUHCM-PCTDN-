# KinAwing-Pose: Anatomically-Constrained Vision Transformers with Hard Keypoint Mining and Model Soup for Surveillance Human Pose Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![WACV 2027 Workshop](https://img.shields.io/badge/WACV%202027-RWS%20Workshop-green.svg)](https://vap.aau.dk/rws/)
[![Codabench Benchmark](https://img.shields.io/badge/AP-0.9001-gold.svg)](https://www.codabench.org/competitions/18273/)

Official source code and technical documentation for **Team VNUHCM-PCTDN** (University of Science, VNU-HCM) in the **UPAR Challenge 2027 (Track 3: 2D Human Pose Estimation under Domain Shift)**, held in conjunction with the **7th Real-World Surveillance (RWS) Workshop at WACV 2027**.

---

## 🏆 Challenge Benchmark Leaderboard (30,245 Test Surveillance Crops)

Evaluated directly on the official CodaBench evaluation server with an Nvidia Tesla T4 GPU:

| Method / Submission | AP | AP50 | AP75 | AP_med | AP_large | AR | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ViTPose-Base (Pretrained, 256x192) | 0.8590 | 0.9502 | 0.9010 | 0.8310 | 0.6420 | 0.8845 | 24.5 min |
| ViTPose-Large (Pretrained, 384x288) | 0.8735 | 0.9580 | 0.9168 | 0.8490 | 0.6610 | 0.8990 | 42.0 min |
| ViTPose-Large (Full Baseline MSE, Ep 4) | 0.8849 | 0.9632 | 0.9270 | 0.8580 | 0.6720 | 0.9080 | 54.7 min |
| ViTPose-Large + KinAwingOHKM (Epoch 2) | 0.8999 | 0.9707 | 0.9369 | **0.8736** | 0.6891 | 0.9186 | 64.2 min |
| ViTPose-Large + KinAwingOHKM (Epoch 4) | 0.8992 | 0.9701 | 0.9360 | 0.8695 | 0.6880 | 0.9180 | 64.2 min |
| **ViTPose-Large + Model Soup (Champion)** | **0.9001** | **0.9705** | **0.9369** | 0.8734 | **0.6895** | **0.9186** | **67.8 min** |

---

## 💡 Key Method Highlights

1. **High-Resolution Vision Transformer Backbone:**
   - **ViTPose-Large** ($384 \times 288$ input, 307M parameters, $24 \times 18 = 432$ visual tokens).
   - Global self-attention captures long-range kinematic dependencies across occluded human limbs.

2. **KinAwingOHKM Multi-Task Compound Loss:**
   - **Adaptive Wing (AWing) Heatmap Loss:** Non-linear exponentially steeper gradients near peak centers prevent gradient starvation.
   - **Targeted Joint Reweighting:** Head keypoints boosted by $2.0\times$ (eliminating downward tilt inversions); wrists boosted by $1.5\times$.
   - **Online Hard Keypoint Mining (OHKM):** Backpropagates through the Top-8 hardest joint losses per instance.
   - **18-Bone Kinematic Length Consistency Regularizer:** Enforces physical skeletal proportions via differentiable soft-argmax 2D coordinates.
   - **Differentiable Soft-Argmax OKS Loss:** Direct metric alignment with competition evaluation criteria.

3. **Stochastic Weight Averaging Model Soup Checkpoint Fusion:**
   - Convex combination of converged weights: $\mathbf{W}_{\text{soup}} = 0.6 \cdot \mathbf{W}_{\text{Ep4}} + 0.4 \cdot \mathbf{W}_{\text{Ep2}}$.
   - Settles into the flattest empirical loss basin to suppress sub-pixel jitter with **strictly 0% added inference latency**.

4. **Robust Inference Pipeline:**
   - Bilateral Flip Test-Time Augmentation (Flip-TTA).
   - Distribution-Aware Coordinate Representation (**DARK**) second-order Hessian Taylor-expansion sub-pixel decoding.
   - Traced standard Float32 TorchScript ensuring 100% deterministic container execution without cuDNN runtime crashes.

---

## 🛠️ Repository Layout

```text
├── assets/
│   ├── mean_pose.json          # Empirical mean keypoint coordinates
│   └── preprocess.json         # Normalization parameters & input dimensions
├── train_vitpose_large_384_stage2_top1.ipynb  # Complete end-to-end training notebook
├── run.py                      # Standalone CodaBench inference pipeline
├── metadata.yaml               # Challenge submission metadata
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/AIVIETNAM-AIO-Koka/UPAR2027-Track3-VNUHCM-PCTDN-.git
cd UPAR2027-Track3-VNUHCM-PCTDN
pip install -r requirements.txt
```

### 2. Download Model Checkpoints

Download the champion TorchScript model and weights from the official drive link:
- **Trained Model Checkpoint (`vitpose.torchscript`, 1.6 GB):** [Google Drive Download Link](https://drive.google.com/drive/folders/your_folder_id)
Place `vitpose.torchscript` inside the `assets/` directory:
```bash
mkdir -p assets
# Place vitpose.torchscript inside assets/
```

### 3. Run Standalone Inference

To evaluate on sample pedestrian crops matching CodaBench's ingestion protocol:
```python
import run

# Run inference on sample image
predictions = run.predict_image(sample_image_dict)
```

### 4. Training from Scratch

To reproduce model training from pre-trained weights:
1. Open and execute the Jupyter Notebook:
   ```bash
   jupyter notebook train_vitpose_large_384_stage2_top1.ipynb
   ```
2. The notebook executes:
   - Data loading from UPAR (PA-100K, Market-1501, PETA)
   - Mixed-precision training with KinAwingOHKM loss
   - Cosine annealing learning rate scheduling
   - Model Soup checkpoint interpolation
   - Float32 TorchScript model export

---

## 📝 Citation

If you find this work helpful for your research, please cite our technical paper:

```bibtex
@inproceedings{tran2027kinawing,
  title={KinAwing-Pose: Anatomically-Constrained Vision Transformers with Hard Keypoint Mining and Stochastic Weight Averaging for Surveillance Human Pose Estimation},
  author={Tran, Khoa and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops},
  year={2027}
}
```

---

## 👥 Authors & Team
- **Team:** VNUHCM-PCTDN
- **Institution:** University of Science, Vietnam National University - Ho Chi Minh City (HCMUS, VNU-HCM)
- **Email:** 2512036546@student.hcmus.edu.vn
