# GNN Prediction of Band Gaps and Formation Energies from 3D Crystal Structures

A reproducible materials-informatics workflow for predicting two density-functional-theory (DFT) properties directly from periodic crystal structures:

- formation energy per atom (`formation_energy_peratom`, eV/atom)
- OptB88-vdW band gap (`optb88vdw_bandgap`, eV)

The project uses real structures and DFT labels from JARVIS-DFT and compares a compact multi-task crystal graph neural network (GNN) against a classical ExtraTrees structural baseline on exactly the same frozen test set.

On the 800-material held-out test set, the CrystalGNN achieves:

- formation-energy MAE: **0.2231 eV/atom**
- formation-energy R²: **0.9001**
- band-gap MAE: **0.3963 eV**
- band-gap R²: **0.6501**

The GNN reduces test MAE relative to ExtraTrees by approximately **34.3% for formation energy** and **21.7% for band gap**.

---

## Project overview

The workflow is:

```text
JARVIS-DFT
    |
    v
Deterministic 8,000-material subset
    |
    +-----------------------------+
    |                             |
    v                             v
Classical structural         Periodic crystal
descriptors                  graph construction
    |                             |
    v                             v
ExtraTrees baseline          Multi-task CrystalGNN
    |                             |
    +-------------+---------------+
                  |
                  v
         Frozen 800-material test set
                  |
                  v
       Model comparison and analysis
                  |
                  v
       Retrospective candidate screen