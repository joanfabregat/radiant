# Technical Architecture

*Implementation details for Project Radiant's clinical workflow platform.*

---

## Proposed Architecture

This architecture is designed to **maximize patient safety, modularity, and regulatory alignment**, by clearly separating detection, contextualization, and report drafting responsibilities.

### High-level pipeline (3 steps, 2 alternatives)

1. **Segmentation (boundary detection)**
2. **Classification or RAG**
3. **LLM / VLM for report generation**

#### Why RAG instead of pure classification

* Can detect:
    * When **no match exists**
    * When **multiple options have similar confidence**
* Safer and more informative in clinical workflows

---

## Step 1 – Segmentation (Boundary Detection)

### Recommended architectures

* **U-Net / U-Net++** – gold standard in medical imaging
* **Mask R-CNN** – instance segmentation (multiple findings)
* **DeepLabV3+** – strong semantic segmentation
* **TransUNet / Swin-UNETR** – transformer-based, long-range context

### Combined segmentation + classification models

* Mask R-CNN
* YOLACT / YOLOv8-seg
* Detectron2-based architectures

---

## Step 2 – Classification or RAG

### Classification models

* ResNet-50 / ResNet-101
* EfficientNet
* DenseNet
* Vision Transformers (ViT)
* Swin Transformer

### RAG: Vectorization options

**General vision encoders**

* ResNet / EfficientNet
* ViT

**Vision–language (CLIP-based)**

* OpenAI CLIP
* BiomedCLIP
* MedCLIP / PubMedCLIP
* MedCLIP, BioCLIP, CheXzero, GLoRIA

**Medical-specific encoders**

* CheXpert / CheXNet
* RadImageNet-pretrained models
* MoCo-CXR / SimCLR-CXR

**Custom approaches**

* Siamese networks / Triplet loss

---

## Empirical Fit of CLIP-style Models on X-rays

CLIP-like models perform well on X-rays due to their 2D structure and paired reports:

* Image–report matching
* Zero-shot / few-shot classification
* Triage (normal vs abnormal, risk prioritization)
* Text-guided search

These models often match or outperform supervised CNN baselines on classification-style tasks.

---

## Step 3 – Report Generation (VLM / LLM)

**Medical VLMs**

* LLaVA-Med
* Med-Flamingo
* CheXagent
* RadFM

**General VLMs**

* GPT-4V
* LLaVA 1.5 / 1.6
* Qwen-VL
* InternVL

**Fine-tuning strategy**

* LoRA / QLoRA
* Inputs: X-rays + segmentation masks
* Outputs: Draft reports (optionally structured)

---

## Pre-training & Tooling

* **ImageNet pre-training transfers well**, even for grayscale X-rays
* Medical-specific pre-training improves performance further

### Platforms & libraries

* MedMNIST
* torchxrayvision
* MONAI
* Hugging Face

---

## Datasets

### Public datasets

* **ChestX-ray14 (NIH)** – 100k+ labeled images
* **CheXpert (Stanford)** – 224k radiographs
* **MIMIC-CXR** – 377k images + reports (ideal for step 3)
* **RSNA Pneumonia Detection** – includes bounding boxes
* **VinDr-CXR** – detailed annotations

---

## Open Questions

1. Does the current dataset include **segmentation annotations**?
2. Which **embedding model(s)** should be used for RAG?
3. Should multiple embedding models and **multiple vector spaces** be used in parallel to capture different characteristics?
