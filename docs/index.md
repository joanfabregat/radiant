# Project Radiant

*A Human-in-the-Loop Clinical Workflow Platform (Radiology-first)*

> A similar project already exists in France: [**Raidium**](https://raidium.eu)

---

## Executive Summary

Project Radiant is a **human-in-the-loop clinical workflow platform** designed to assist physicians by streamlining detection, qualification, and reporting tasks—starting with **radiology (X-ray)** and extending to **dermatology and other specialties**. The company deliberately avoids automated diagnosis, focusing instead on **workflow optimization, triage, quality assurance, and report drafting assistance**.

The core innovation lies not in model novelty but in a **regulator-safe system architecture** aligned with **EU MDR requirements** from day one. The platform separates responsibilities across detection, contextual retrieval, and report drafting, ensuring that all outputs remain assistive, traceable, and subject to clinician validation.

Project Radiant's go-to-market strategy is anchored in **French teleradiology**, providing rapid access to real clinical workflows, early validation, and a strong regulatory foundation. Radiology serves as the first vertical and proof point; subsequent specialties reuse the same process, governance, and infrastructure.

The company is positioned as **clinical workflow infrastructure with embedded AI**, not an AI diagnostics startup. Its defensibility is built on **workflow integration, regulatory rigor, clinical trust, and ROI**, supported by a US–France operating model optimized for innovation, compliance, and fundraising.

---

## Vision & Positioning

Project Radiant aims to build **AI-assisted clinical workflow software** to support physicians, starting with **radiology (X-ray)** and later expanding to **dermatology and other specialties**.

### Key positioning

* Not an "AI diagnosis" company
* Focused on:
    * Workflow optimization
    * Triage & prioritization
    * Quality assurance
    * Report drafting assistance
* Competitive advantage comes from:
    * Regulatory discipline (EU MDR)
    * Clinical integration
    * Trust, auditability, and ROI

Radiology is the **first vertical**, not the end goal.

---

## Core Technical Abstraction

What generalizes across specialties is **process**, not models.

### Generalized workflow

1. **Detect** an abnormal signal
2. **Qualify** it using embeddings + vector search
3. **Draft** a medical report using an LLM

### Key principle

* Detection models are **specialty-specific**
* Vector search and LLMs must remain:
    * Assistive
    * Constrained
    * Human-in-the-loop

---

## Conclusion

Project Radiant is a **human-in-the-loop clinical workflow platform**, built to EU regulatory standards from day one. Its defensibility lies not in model novelty, but in **workflow integration, regulatory rigor, and clinical trust**.

Radiology serves as the initial proof point; dermatology and future specialties extend through the same process-driven architecture.

---

## Documentation

- [Technical Architecture](./technical.md) — Architecture, models, datasets, and implementation details
- [Business & Strategy](./business.md) — Market entry, regulatory strategy, fundraising, and operating model
