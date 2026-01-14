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
                          │            │  Method: RAG → Rerank                │
                          │            └──────────────────────────────────────┘
                          │                         │
                          └────────────┬────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Report Generation (two-stage)                                      │
│  ─────────────────────────────────────                                      │
│  Stage A: VLM describes each cropped anomaly                                │
│  Stage B: LLM synthesizes findings into coherent report                     │
│                                                                             │
│  Input:  - Cropped anomaly images (NOT full X-ray)                          │
│          - Qualification results from Step 2                                │
│          - Location metadata                                                │
│  Output: Draft report (structured text)                                     │
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

### Training datasets for Step 1

| Dataset | Size | Notes |
|---------|------|-------|
| RSNA Pneumonia | 30k images | Bounding box annotations — ideal for detection training |
| VinDr-CXR | 18k images | Detailed local annotations with bounding boxes |

---

## Step 2 – Anomaly Qualification

**Goal:** For each detected anomaly, determine what it is by comparing to known examples.

### Input / Output

| | |
|---|---|
| **Input** | Cropped image region for each bounding box from Step 1 |
| **Output** | For each anomaly: top-k similar cases, similarity scores, suggested label(s) |

### Two-stage retrieval pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Cropped        │     │  Stage 1:       │     │  Stage 2:       │
│  anomaly        │ ──▶ │  Embedding      │ ──▶ │  Reranker       │ ──▶ Final results
│  image          │     │  retrieval      │     │  (image-to-     │
│                 │     │  (top-k, fast)  │     │  image)         │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Stage 1: Embedding retrieval (RAG)

Embed each cropped anomaly and search against a vector database of known anomaly examples.

**Why RAG is preferred:**
* Can signal "no confident match" (important for safety)
* Interpretable: show clinician similar cases
* Extensible: add new anomaly types without retraining
* Aligns with regulatory "retrieval, not reasoning" framing

**Embedding model**

Use a single medical CLIP-style model. Candidates to evaluate:

| Model | Notes |
|-------|-------|
| BiomedCLIP | Medical vision-language model, good baseline |
| CheXzero | Trained on chest X-rays specifically |
| MedCLIP / PubMedCLIP | Medical CLIP variants |
| RadImageNet-pretrained | Domain-specific pretraining |

**Nice to have:** Multiple embedding models in parallel (capturing different aspects: texture, shape, semantic meaning) with a synthetic score combining results. Only pursue if single-model evaluation reveals clear blind spots.

**Vector database**
* Corpus: Embeddings of annotated anomaly examples (internal curated dataset, in progress)
* Metadata per entry: anomaly label, source image ID, anatomical location
* Retrieval: top-k nearest neighbors (e.g., k=50) with similarity scores

### Stage 2: Visual reranking

After embedding retrieval, a reranker compares the query crop directly with retrieved example images to refine the ordering.

**Why rerank:**
* Embeddings capture high-level similarity; reranking verifies fine-grained visual match
* Catches cases where embeddings are similar but actual appearance differs
* Improves precision at the top of the results

### Image-to-image comparison

The reranker compares the query crop visually with each retrieved example image. This is the core reranking approach because the goal is to verify visual similarity.

**Summary:**

| Question | Answer |
|----------|--------|
| Can an existing model be used? | Yes. Use the same BiomedCLIP model from Stage 1 — essentially free since it's already loaded. |
| Does it need fine-tuning? | No for baseline. For better precision, a Siamese network can be trained. |
| Does a new model need to be created? | No. Reuse BiomedCLIP. Siamese network is optional enhancement. |
| Dataset required | None for zero-shot. Siamese needs pairs of crops labeled "same type" or "different type." |

**Default approach: Zero-shot CLIP reranking**

Since BiomedCLIP is already used for Stage 1 embeddings, reranking is essentially free:
* Compute embeddings for query crop and each retrieved example
* Compare with cosine similarity
* Reorder results by similarity score
* No additional model, no training data needed

**If needed: Fine-tuned Siamese network**

If zero-shot reranking isn't precise enough, train a Siamese network:
* Uses pretrained backbone (ResNet, ViT) + learned comparison head
* Dataset: pairs of anomaly crops labeled "same type" or "different type"
* Construct from internal dataset: same-label = positive pair, different-label = negative pair
* Include hard negatives (visually similar but different diagnosis)

**Model options for Siamese (if needed)**

| Model | Notes |
|-------|-------|
| Fine-tuned Siamese network | Train on internal dataset to learn "same anomaly type" vs "different." Most controllable. |
| DINOv2 + comparison head | Strong self-supervised features, good at fine-grained visual similarity |
| Learned metric networks | ProtoNet, Matching Networks — designed for similarity learning |

### Nice to have: Image-to-label comparison

As an additional confidence signal, compare the query crop against text labels: "does this crop look like a typical [suggested label]?"

**Summary:**

| Question | Answer |
|----------|--------|
| Can an existing model be used? | Yes, directly. This is exactly what CLIP-style models do. BiomedCLIP or CheXzero work out of the box. |
| Does it need fine-tuning? | For baseline: No. For better alignment with your label vocabulary: light fine-tuning helps. |
| Does a new model need to be created? | No. This is native CLIP functionality. |
| Dataset required | Zero-shot: Just the label vocabulary. Fine-tuning: anomaly crops paired with text labels. |

**How CLIP image-to-text similarity works:**

CLIP is trained to understand images and text in the **same embedding space**. Both get converted to vectors that can be compared directly.

```
┌─────────────────┐                              ┌─────────────────┐
│  Cropped        │                              │  Text:          │
│  anomaly image  │                              │  "nodule"       │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│  CLIP Image     │                              │  CLIP Text      │
│  Encoder        │                              │  Encoder        │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
   [0.1, 0.3, -0.2, ...]                          [0.15, 0.28, -0.18, ...]
   (image embedding)                              (text embedding)
         │                                                │
         └──────────────────┬─────────────────────────────┘
                            │
                            ▼
                   Cosine similarity = 0.85
                   (high = image matches text)
```

Compare one image against multiple labels to find the best match:

```
similarity(crop, "nodule")        → 0.85  ← highest = likely a nodule
similarity(crop, "consolidation") → 0.30
similarity(crop, "fracture")      → 0.10
```

**Why this is useful:**
* No classifier training needed
* Add new labels by just adding new text strings
* "Zero-shot" — works without training examples of that specific label

BiomedCLIP and CheXzero understand medical terms like "nodule," "consolidation," "cardiomegaly."

**Implementation options:**

| Approach | New model? | Training data? | Notes |
|----------|------------|----------------|-------|
| Zero-shot CLIP | No | No | Use pretrained BiomedCLIP/CheXzero directly. Just need label vocabulary. |
| Fine-tuned CLIP | No | Light | Fine-tune on your label vocabulary for better alignment. |

**Why nice to have:** This is low-effort if already using CLIP for embeddings, but adds complexity to the pipeline. Validate core RAG + reranking first.

### Fallback: Classification (Plan B)

If embeddings do not produce good clustering of similar anomalies, fall back to a trained classifier.

* Input: Cropped anomaly region
* Output: Class label + confidence score
* Models: ResNet, EfficientNet, ViT, Swin Transformer

**Downsides vs RAG:**
* Cannot express "I don't know"
* Requires retraining to add new classes
* Less interpretable

**Training datasets for classification fallback:**

| Dataset | Size | Notes |
|---------|------|-------|
| ChestX-ray14 (NIH) | 112k images | 14 disease labels |
| CheXpert (Stanford) | 224k images | Uncertainty labels, multi-label |

### Multi-anomaly handling

If Step 1 detects multiple anomalies (e.g., 3 bounding boxes), each is processed through Step 2 **independently**. All results are passed together to Step 3.

---

## Step 3 – Report Generation

**Goal:** Generate a draft radiology report summarizing all findings (or lack thereof).

### Design principle: Prevent late detection

The VLM receives **only cropped anomaly images**, not the full X-ray. This prevents the model from "discovering" new findings not detected in Step 1. The report is strictly based on what the detection pipeline found.

### Input / Output

| | |
|---|---|
| **Input** | Cropped anomaly images, qualification results from Step 2, location metadata |
| **Output** | Structured draft report |

### Two-stage approach

**Stage A: VLM describes each anomaly**

For each detected anomaly, a VLM receives:
1. The cropped anomaly image
2. Similar cases from Step 2 (for context)
3. Suggested labels and confidence scores
4. Location metadata (e.g., "right lower lobe")

It outputs a description of that specific finding.

**Stage B: LLM synthesizes report**

An LLM receives all individual finding descriptions and synthesizes them into a coherent radiology report following a structured schema (e.g., Findings, Impressions sections).

**Why two-stage:**
* Stage A keeps vision isolated to individual findings
* Stage B is text-to-text, easier to control and audit
* Clear separation of responsibilities

### Nice to have: Redacted X-ray input

Instead of crops only, provide the full X-ray with:
* Bounding boxes drawn on detected anomalies
* Areas outside bounding boxes redacted/grayed out

This gives spatial context while still preventing the VLM from seeing undetected regions.

### Model options

**Stage A (VLM for crop description)**

| Model | Notes |
|-------|-------|
| GPT-4o / Claude | Strong general VLMs with medical prompting |
| CheXagent | Trained on chest X-ray reports (may need adaptation for crops) |
| LLaVA-Med | Medical fine-tuned, open-source |
| RadFM | Radiology foundation model |

Note: Most medical VLMs are trained on full image → full report. For crop → description, general VLMs with strong prompting may perform comparably.

**Stage B (LLM for synthesis)**

* GPT-4o / Claude — strong at structured synthesis
* Open-source: Llama 3, Mistral with medical fine-tuning

**Fine-tuning strategy**
* LoRA / QLoRA for efficient adaptation

**Training dataset for Step 3:**

| Dataset | Size | Notes |
|---------|------|-------|
| MIMIC-CXR | 377k images | Includes free-text reports — ideal for report generation training |

### Normal case

If no anomalies were detected in Step 1, skip Stage A. Stage B generates a report indicating a normal scan (e.g., "No acute cardiopulmonary abnormality identified").

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

## Open Questions

1. Which **embedding model** (BiomedCLIP, CheXzero, etc.) produces the best clustering for anomaly crops? (key experiment for RAG viability)
2. What **confidence thresholds** should trigger "low confidence" warnings to clinicians?
3. What is the **report schema** for Step 3 output?
4. Is zero-shot CLIP reranking sufficient, or is a **Siamese network** needed for better precision?
