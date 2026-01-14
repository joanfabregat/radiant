# Technical Architecture

*Implementation details for Project Radiant's clinical workflow platform.*

---

## Pipeline Overview

The architecture separates detection, qualification, and report generation into three distinct steps for **observability, traceability, and regulatory alignment**.

### High-level pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT: X-ray image                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Anomaly Detection                                                  │
│  ─────────────────────────                                                  │
│  Input:  Full X-ray image                                                   │
│  Output: List of bounding boxes around detected anomalies                   │
│          (empty list if no anomalies found)                                 │
│  Model:  Object detection (e.g., YOLOv8, Mask R-CNN, DETR)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          │                         │
                    No anomalies              Anomalies found
                          │                         │
                          ▼                         ▼
              ┌───────────────────┐    ┌──────────────────────────────────────┐
              │ Skip to Step 3    │    │  STEP 2: Anomaly Qualification       │
              │ (normal report)   │    │  ───────────────────────────         │
              └───────────────────┘    │  Input:  Cropped anomaly regions     │
                          │            │  Output: For each anomaly:           │
                          │            │          - Top-k similar cases       │
                          │            │          - Similarity scores         │
                          │            │          - Suggested labels          │
                          │            │  Method: RAG (Plan A) or             │
                          │            │          Classification (Plan B)     │
                          │            └──────────────────────────────────────┘
                          │                         │
                          └────────────┬────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Report Generation                                                  │
│  ─────────────────────────                                                  │
│  Input:  - Original X-ray                                                   │
│          - Bounding boxes from Step 1                                       │
│          - Qualification results from Step 2 (if any)                       │
│  Output: Draft report (structured text)                                     │
│  Model:  LLM or VLM with constrained output schema                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLINICIAN REVIEW                                                           │
│  ─────────────────                                                          │
│  - View detected anomalies with bounding boxes                              │
│  - Review qualification results (similar cases)                             │
│  - Edit draft report: confirm, remove, or add findings                      │
│  - Added findings feed back into training data                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1 – Anomaly Detection

**Goal:** Locate abnormal regions in the X-ray. Output bounding boxes around each detected anomaly.

### Input / Output

| | |
|---|---|
| **Input** | Full X-ray image (DICOM or PNG) |
| **Output** | List of bounding boxes `[(x, y, w, h, confidence), ...]` — empty if no anomalies |

### Recommended architectures

**Object detection (preferred)**

* **YOLOv8** – fast, accurate, good for real-time
* **DETR / RT-DETR** – transformer-based, no anchor tuning
* **Faster R-CNN** – well-established, good accuracy

**Instance segmentation (if pixel-level masks needed)**

* **Mask R-CNN** – instance segmentation with bounding boxes
* **YOLOv8-seg** – fast segmentation variant

### Normal case handling

If Step 1 detects **no anomalies**, the pipeline skips Step 2 and proceeds directly to Step 3 with an empty finding list. Step 3 generates a report indicating no abnormalities were detected.

---

## Step 2 – Anomaly Qualification

**Goal:** For each detected anomaly, determine what it is by comparing to known examples.

### Input / Output

| | |
|---|---|
| **Input** | Cropped image region for each bounding box from Step 1 |
| **Output** | For each anomaly: top-k similar cases, similarity scores, suggested label(s) |

### Plan A: RAG (Retrieval-Augmented Qualification)

Embed each cropped anomaly and search against a vector database of known anomaly examples.

**Why RAG is preferred:**
* Can signal "no confident match" (important for safety)
* Interpretable: show clinician similar cases
* Extensible: add new anomaly types without retraining
* Aligns with regulatory "retrieval, not reasoning" framing

**Embedding options (to evaluate)**

| Model | Notes |
|-------|-------|
| BiomedCLIP | Medical vision-language model, good baseline |
| CheXzero | Trained on chest X-rays specifically |
| MedCLIP / PubMedCLIP | Medical CLIP variants |
| RadImageNet-pretrained | Domain-specific pretraining |

**Vector database**
* Corpus: Embeddings of annotated anomaly examples (curated dataset)
* Metadata per entry: anomaly label, source image ID, anatomical location
* Retrieval: top-k nearest neighbors with similarity scores

**Open question:** Should multiple embedding models be used in parallel (multiple vector spaces) with a synthetic score combining their results?

### Plan B: Classification

If embeddings do not produce good clustering of similar anomalies, fall back to a trained classifier.

* Input: Cropped anomaly region
* Output: Class label + confidence score
* Models: ResNet, EfficientNet, ViT, Swin Transformer

**Downsides vs RAG:**
* Cannot express "I don't know"
* Requires retraining to add new classes
* Less interpretable

### Multi-anomaly handling

If Step 1 detects multiple anomalies (e.g., 3 bounding boxes), each is processed through Step 2 **independently**. All results are passed together to Step 3.

---

## Step 3 – Report Generation

**Goal:** Generate a draft radiology report summarizing all findings (or lack thereof).

### Input / Output

| | |
|---|---|
| **Input** | Original X-ray, bounding boxes, qualification results for each anomaly |
| **Output** | Structured draft report |

### Approach

The LLM/VLM receives:
1. The original X-ray image (or a summary representation)
2. Detected anomalies with locations and suggested labels
3. Qualification context (similar cases, confidence levels)

It produces a draft report following a constrained schema (e.g., Findings, Impressions sections).

**Key constraints:**
* LLM must not introduce findings not present in Step 1/2 outputs
* Output follows a strict template/schema
* All findings are traceable to specific bounding boxes

### Model options

**Medical VLMs**
* CheXagent – trained on chest X-ray reports
* RadFM – radiology foundation model
* LLaVA-Med – medical fine-tuned LLaVA

**General VLMs (with fine-tuning)**
* GPT-4o / Claude – strong general reasoning
* LLaVA 1.5/1.6, Qwen-VL, InternVL – open-source options

**Fine-tuning strategy**
* LoRA / QLoRA for efficient adaptation
* Training data: MIMIC-CXR (images + reports)

### Normal case

If no anomalies were detected in Step 1, Step 3 generates a report indicating a normal scan (e.g., "No acute cardiopulmonary abnormality identified").

---

## Human-in-the-Loop Integration

The clinician receives the complete pipeline output:

1. **Original X-ray** with bounding boxes overlaid
2. **Per-anomaly detail view**: cropped region, similar cases from RAG, suggested label
3. **Draft report** ready for review

**Clinician actions:**
* **Confirm** findings as-is
* **Remove** false positive detections
* **Add** missed findings (these are logged and fed back to improve models)
* **Edit** report text

All clinician modifications are logged for model improvement and audit.

---

## Datasets

### Public datasets

| Dataset | Size | Notes |
|---------|------|-------|
| ChestX-ray14 (NIH) | 112k images | 14 disease labels |
| CheXpert (Stanford) | 224k images | Uncertainty labels |
| MIMIC-CXR | 377k images | Includes free-text reports (ideal for Step 3) |
| RSNA Pneumonia | 30k images | Bounding box annotations (ideal for Step 1) |
| VinDr-CXR | 18k images | Detailed local annotations |

### Internal dataset (in progress)

Curated anomaly examples for the RAG corpus (Step 2).

---

## Pre-training & Tooling

* **ImageNet pre-training** transfers well to X-rays
* **Medical-specific pre-training** (RadImageNet, CheXpert) improves performance

### Libraries

* MONAI – medical imaging toolkit
* torchxrayvision – pretrained X-ray models
* Hugging Face – model hub and training utilities

---

## Open Questions

1. Does the current dataset include **bounding box annotations** for Step 1 training?
2. Which **embedding model(s)** produce the best clustering for anomaly crops? (key experiment for RAG viability)
3. Should multiple embedding models and vector spaces be used with a **synthetic scoring** approach?
4. What **confidence thresholds** should trigger "low confidence" warnings to clinicians?
5. What is the **report schema** for Step 3 output?
