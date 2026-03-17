# LHCO CWoLa Anomaly Detection

A weakly supervised anomaly detection project on the **LHC Olympics 2020 R&D dataset** using **Classification Without Labels (CWoLa)** to identify signal-like collision events and enhance resonance discovery sensitivity.

## Overview

This project implements a full end-to-end collider anomaly detection pipeline using the LHCO dataset. The method is based on **CWoLa**, where events from a signal region and sidebands are used as pseudo-labels to train a classifier without direct signal supervision.

The pipeline includes:

- downloading the LHCO dataset from Zenodo
- loading and inspecting the large HDF5 event table
- engineering physics-inspired jet and dijet features
- defining signal and sideband regions in dijet invariant mass
- training a dense neural CWoLa classifier
- evaluating pseudo-label and truth-label ROC performance
- ranking top signal-like events
- producing physics plots and signal-enrichment tables

## Dataset

- **Dataset:** LHC Olympics 2020 R&D dataset
- **Source:** Zenodo
- **Events:** ~1.1 million
- **Composition:** QCD background with injected signal events
- **Main file used:** `events_anomalydetection.h5`

## Features Used

Input data contains high-level two-jet observables:

- `pxj1, pyj1, pzj1, mj1`
- `tau1j1, tau2j1, tau3j1`
- `pxj2, pyj2, pzj2, mj2`
- `tau1j2, tau2j2, tau3j2`
- `label`

Derived features include:

- `ptj1, ptj2`
- `Ej1, Ej2`
- `mjj`
- `tau21_j1, tau32_j1`
- `tau21_j2, tau32_j2`
- `pt_balance`
- `m_ratio`

## Methodology

### CWoLa strategy

- define a **signal region (SR)** in dijet invariant mass
- define **left and right sidebands (SB)**
- assign pseudo-labels:
  - SR → 1
  - SB → 0
- exclude `mjj` from training inputs
- train a classifier to distinguish SR from SB
- use classifier score as anomaly relevance score

### Model

A dense neural network classifier with:

- fully connected layers
- batch normalization
- dropout regularization
- binary cross-entropy loss
- AUC tracking during training

## Outputs

The notebook generates:

training loss and AUC curves

pseudo-label ROC curve

truth-label ROC curve

CWoLa score distributions

score vs dijet mass plots

post-selection mass distributions

signal-enrichment tables

top candidate event rankings

## Physics Motivation

This project explores weakly supervised anomaly detection for new physics searches at the LHC. Rather than training directly on signal labels, the classifier learns differences between signal-region and sideband events. If the learned score aligns with hidden truth labels, it indicates that the model is capturing anomaly-relevant physical structure.

## Future Extensions

XGBoost baseline comparison

k-fold CWoLa

decorrelated training

particle-level models

bump hunting after anomaly selection

comparison with autoencoders and VAEs

License

This project is released under the MIT License.

