# 🧠 Brain Tumor Classification using Vision Transformer (ViT)

A deep learning-based solution for classifying brain tumors from MRI scans using a Vision Transformer (ViT-B16) architecture implemented in PyTorch. This repository aims to leverage the power of transformers in computer vision for accurate and interpretable brain tumor detection.

![Vision Transformer MRI Classification](https://img.shields.io/badge/Model-ViT--B16-blue)
![License](https://img.shields.io/github/license/yourusername/brain-tumor-vit)
![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Training & Evaluation](#training--evaluation)
- [Visualizations](#visualizations)
- [Conclusion](#future-work)
---

## 🧩 Overview

This project focuses on building a Vision Transformer model from scratch to classify brain tumors into their respective categories using MRI images. The goal is to explore transformer-based models in medical imaging and compare their performance with traditional CNNs.

---

## 📂 Dataset

We used a publicly available dataset:

- **Name**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes**: `Glioma`, `Meningioma`, `Pituitary`, `No Tumor`
- **Format**: Pre-labeled MRI images
---
![download](https://github.com/user-attachments/assets/4c26d7b3-c786-42a8-ae74-c79a7afa1bf6)


## 🧠 Model Architecture

Our implementation is based on **ViT-B16 (Vision Transformer)**, including:

- Patch Embedding
- Positional Encoding
- Transformer Encoder Blocks
- MLP Head for Classification

Custom modules include:
- `PatchEmbedding`
- `TransformerEncoderLayer`
- `ClassificationHead`

> **Framework**: PyTorch  
> **Input size**: 224x224 RGB  
> **Optimizer**: AdamW  
> **Loss Function**: CrossEntropyLoss  

---

## ⚙️ Installation

```bash
git clone [https://github.com/yourusername/brain-tumor-vit.git](https://github.com/GUNNER2K/Vision_Transformers-For-Brain-tumor-classification.)
pip install -r requirements.txt
python ViT_From_scratch/train.py
```

## ⚙️ Training and Evaluation
![image](https://github.com/user-attachments/assets/a30a5973-6454-455b-aded-960a6ecd38da)

![acc_plot_VIT_multi](https://github.com/user-attachments/assets/38701226-8190-42d8-86d4-21006dcea528)


## ⚙️ Visualization

![Vit_multi_confusion_matrix](https://github.com/user-attachments/assets/9f3b624f-fa8f-4396-aa88-f8031153539d)

## 📚 References

- **E. Simon, A. Briassouli** (2022) - _On the performance of Vision Transformers for brain tumor classification_
- **Wang et al., 2021** - _Convolutional Vision Transformers (CvT): A unified architecture for image classification_

---

## 📝 Conclusion

Vision Transformers hold promising potential in medical imaging, especially in tasks like tumor classification. However, their reliance on large-scale data makes them challenging in limited-data scenarios. Future directions include hybrid architectures like CvTs and the use of transfer learning with domain-specific pretraining.


