# GNN Prediction of Band Gaps and Formation Energies from 3D Crystal Structures

Graph neural network prediction of DFT materials properties directly from periodic 3D crystal structures.

## Dataset

This project uses the JARVIS-DFT `dft_3d` dataset.

Primary targets:

- Formation energy per atom (`formation_energy_peratom`)
- OptB88-vdW band gap (`optb88vdw_bandgap`)

A deterministic subset of 8,000 materials is used:

- Training: 6,400
- Validation: 800
- Test: 800
- Random seed: 42

## Methods

The project compares:

- Classical structural descriptors with ExtraTrees regression
- Periodic crystal graph neural networks

Crystal graphs use periodic-image neighbor enumeration and radial distance features.

## Current Baseline Test Results

Formation energy:

- MAE: 0.3396 eV/atom
- RMSE: 0.4904 eV/atom
- R²: 0.7882

Band gap:

- MAE: 0.5058 eV
- RMSE: 0.9169 eV
- R²: 0.5377

## Data Source

JARVIS-DFT:

- https://doi.org/10.1016/j.commatsci.2025.114063
- https://doi.org/10.6084/m9.figshare.6815699

## Status

In development.