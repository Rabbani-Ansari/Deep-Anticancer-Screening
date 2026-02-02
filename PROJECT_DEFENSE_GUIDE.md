# 🎓 OncoScreen AI: Project Defense & Technical Guide

This document is specifically prepared for your **Project Defense / Viva** with your mentor or teacher. It explains the "Why" and "How" of every component in the OncoScreen project.

---

## 1. Project Identity
*   **Name**: OncoScreen AI (formerly DrugDiscover)
*   **Domain**: Computational Oncology & AI-Driven Drug Discovery.
*   **Objective**: To shortlist potential anticancer drug candidates using Graph Neural Networks (GNN) to predict their effectiveness against tumor growth.
*   **Meaning of Name**: **Onco-** (Cancer) + **Screen** (Virtual Screening).

---

## 2. Technical Architecture
The project follows a modern, decoupled architecture:
*   **Frontend**: React (Vite) + Tailwind CSS + Recharts (for analytics).
    *   *Why?*: High performance, responsive UI, and professional data visualization.
*   **Backend**: FastAPI (Python).
    *   *Why?*: Extremely fast, asynchronous, and handles heavy AI computation efficiently.
*   **AI Engine**: PyTorch Geometric (GNN) + RDKit (Chemoinformatics).
    *   *Why?*: Industry standards for molecular graph processing.

---

## 3. The "Brain": Graph Neural Networks (GNN)
**Crucial Point for Teachers**: Unlike traditional AI that uses "Fingerprints" (fixed text-like codes), our GNN looks at the **actual shape** of the molecule.
*   **Nodes**: Atoms (Carbon, Oxygen, etc.).
*   **Edges**: Chemical Bonds (Single, Double, etc.).
*   **Feature Engineering**: 36 atom-level features and 13 bond-level features are processed through message-passing layers.

---

## 4. Data Legitimacy
*   **Source**: The model is trained on **Real-World Scientific Data** from the **ChEMBL** database and the **NCI-60** (National Cancer Institute) specialized cancer datasets.
*   **Quantity**: Integrated over 1,000 verified anticancer drugs and thousands of control molecules.
*   **Target Metric**: **GI50** (50% Growth Inhibition). This is the concentration required to inhibit 50% of cancer cell growth.

---

## 5. Frequently Asked Questions (Defense Prep)

### Q1: Why use a GNN instead of a standard Neural Network (ANN)?
> **Answer**: Standard ANNs require molecules to be converted into flat text strings (SMILES) or bit-vectors (Fingerprints), which loses spatial and structural information. GNNs treat molecules as **graphs**, preserving the connectivity and 3D-like relationships between atoms, leading to higher accuracy in anticancer predictions.

### Q2: What are "Physicochemical Properties" and why do we show them?
> **Answer**: Even if a drug is "Active" against cancer, it might not be a good drug if it can't be absorbed by the body. We calculate **Molecular Weight, LogP (Lipophilicity), and TPSA (Polarity)** to check if the molecule obeys **Lipinski's Rule of Five**, which ensures the drug is "Drug-like" and safe for human consumption.

### Q3: How is the 2D structure generated in the UI?
> **Answer**: We use the **RDKit library** in the backend. It takes the SMILES string (a chemical code), builds a mathematical representation of the molecule, and renders a high-quality SVG/PNG image for the user to inspect.

### Q4: What does "90% Confidence" actually mean in your Screening?
> **Answer**: It represents the statistical probability calculated by the model's Sigmoid output. A 90% score means the model is highly certain, based on its training with historical NCI-60 data, that this specific molecular structure will successfully inhibit tumor growth.

### Q5: Can this system be used for other diseases (like COVID-19 or Flu)?
> **Answer**: Currently, **no**. I have specialized this interface and model specifically for **Oncology (Cancer)**. The model "knows" what cancer cell-line inhibitors look like. To use it for other diseases, we would need to retrain the GNN on a different dataset (e.g., antiviral data).

---

## 6. Key Features Summary
1.  **Single-Molecule Analysis**: Detailed "Quick View" of structural properties and drug-likeness.
2.  **Library Batch Screening**: Ability to upload CSV files with thousands of molecules.
3.  **Real-time Analytics**: Score distribution charts and average hit rates for researcher insight.
4.  **Bioavailability Radar**: Visual representation of Lipinski compliance.

---
*Created by: Rabbani Ansari*
*Purpose: Project Defense / Academic Submission (Feb 2026)*
