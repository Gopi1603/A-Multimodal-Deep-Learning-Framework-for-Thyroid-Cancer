# A Multimodal Deep Learning Framework for Thyroid Cancer Diagnosis

**Team Members**  
- M. Gopi Chakradhar (121CS0050)  
- K Rohith (121CS0045)  

**Guide**  
Dr. N. Srinivas Naik, Department of Computer Science & Engineering, IIITDM Kurnool

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Directory Structure](#directory-structure)  
4. [Data Preparation](#data-preparation)  
5. [Preprocessing Pipelines](#preprocessing-pipelines)  
6. [Model Training](#model-training)  
7. [NLP & Evaluation](#nlp--evaluation)  
8. [Plagiarism Reports](#plagiarism-reports)  
9. [License](#license)

---

## Project Overview

This project presents an end-to-end deep learning pipeline that fuses ultrasound imaging and clinical data for automated thyroid nodule classification and malignancy risk scoring. A BART-based NLP module then generates a concise, human-readable diagnostic report.

## Key Features

- **Hybrid Imaging Model**: CNN + Vision Transformer (ViT) for local and global feature extraction  
- **Clinical MLP**: Processes 18 patient features (age, TSH, FT3, FT4, etc.) into a low-dimensional embedding  
- **Dynamic Fusion**: Learnable gating to adaptively weight imaging vs. clinical signals  
- **Multi-Task Output**:  
  - **Classification** of nodule type (9 classes)  
  - **Regression** of malignancy risk score (0–1)  
- **Automated Reporting**: BART summarizer generates JSON-formatted diagnostic reports  
- **Lightweight Deployment**: Optimized for CPU inference with minimal hardware requirements  

## Directory Structure

```text
├── data
│   ├── external
│   │   └── thyroid_clean.csv
│   ├── processed
│   │   ├── processed_dataset2.zip
│   │   └── processed_dataset3.zip
│   └── raw
│       ├── clinical/thyroid_clean.csv
│       ├── Classic Papillary Thyroid Cancer.v1i.yolov9.zip
│       └── Follicular Variant Thyroid CA.v1i.yolov9.zip
│
├── docs
│   └── images
│       ├── C3.jpg   C5.jpg   D0.jpg
│       ├── fusion.jpg   multi.jpg
│       └── output.png
│
├── nlp & evaluation
│   ├── TESTING-ANALYSIS.ipynb
│   └── nlp-summarization.ipynb
│
├── plagiarism
│   ├── ithenticate.pdf
│   └── turnit.pdf
│
├── ppt
│   ├── 121CS0045_121CS0050_PROJECT_PPT.pdf
│   └── thesis.pdf
│
├── src
│   ├── CLINICAL-data-preprocessing/
│   │   └── preprocess-clinical.ipynb
│   ├── IMAGE-data-preprocessing/
│   │   ├── dataset1-preprocessing-1.pdf
│   │   ├── dataset2-preprocessing.pdf
│   │   ├── dataset3-preprocessing.pdf
│   │   └── *.ipynb
│   ├── models/
│   │   ├── clinical_model.py
│   │   ├── cnn.py
│   │   ├── fusion.py
│   │   ├── hybrid_model.py
│   │   ├── multitask.py
│   │   └── vit.py
│   └── training/
│       ├── train_imaging.py
│       ├── train-imaging-py-d1.ipynb
│       ├── train-imaging-py-d2.ipynb
│       └── train-imaging-py-d3.ipynb
│
├── main.py
├── LICENSE
└── README.md
