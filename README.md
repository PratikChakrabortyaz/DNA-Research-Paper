# An Explainable AI Framework Integrating Machine and Deep Learning Models for Multi-Species DNA Functional Group Classification
This repository contains the complete implementation for a multi-species DNA functional group classification framework developed as part of a research study. The project integrates machine learning, deep learning, and multi-level explainable AI (XAI) techniques to classify gene families and extract biologically meaningful sequence motifs.

## Overview

DNA sequences from Human, Chimpanzee, Dog, and a combined multi-species dataset are represented using k-mers and evaluated across multiple models. The objective is to benchmark predictive performance and derive interpretable motif-level insights grounded in biological relevance.

The repository includes:
- End-to-end model training and evaluation workflows
- Extensive hyperparameter tuning
- Robustness and statistical significance analysis
- Multi-level XAI analysis to identify consensus motifs, model stability, and model fidelity

## Repository Structure

```
notebooks/
│
├── dna-sequence-classification-ml-models.ipynb
│   Contains ML-based classification pipelines (Logistic Regression, Random Forest, Multinomial Naive Bayes),
│   preprocessing steps, hyperparameter tuning, evaluation, and ML-specific XAI analysis.
│
├── dna-sequence-classification-dl-models.ipynb
│   Implements deep learning architectures including CNN, CNN-Attention, and CNN-BiLSTM,
│   along with model training, hyperparameter optimization, and interpretability methods such as Saliency Maps,
│   Integrated Gradients, and GradientSHAP.
│
└── dna-sequence-classification-xai-analysis.ipynb
    Dedicated notebook for multi-level XAI evaluation including:
    - Consensus motif identification across models, datasets, and XAI methods
    - Cross-dataset and cross-model motif comparison
    - Model stability analysis using motif overlap and Jaccard similarity
    - Model fidelity analysis via performance drop after motif masking
```

## Functional Groups (Gene Families)

The classification task involves seven biologically significant gene families:
- G protein-coupled receptors (GPCRs)
- Tyrosine kinases
- Tyrosine phosphatases
- Synthetases
- Synthases
- Ion channels
- Transcription factors

These gene families contain conserved sequence signatures that the models learn to recognize through k-mer representations.

## Multi-Level Explainable AI Framework

The XAI analysis includes:
- Feature importance analysis for ML models
- Saliency Maps, Integrated Gradients, and GradientSHAP for DL models
- Attention-based motif visualization
- Consensus motif extraction across species and models
- Stability analysis (overlap and Jaccard similarity among model-pair motifs)
- Fidelity analysis (performance degradation after masking important motifs)

This framework bridges predictive performance with biological interpretability, identifying motifs aligned with known regulatory regions, catalytic domains, and promoter structures.

## Key Findings

- Logistic Regression with tuned hyperparameters delivered the highest overall accuracy across species.
- Deep learning models captured long-range motif interactions but were more sensitive to dataset size.
- Multi-level XAI revealed stable, recurring motifs linked to known biological functions such as kinase P-loops,
  GPCR transmembrane signatures, and transcription factor promoter elements.
- Fidelity analysis demonstrated that classical ML models rely more heavily on specific motifs, whereas DL models
  exhibit robustness due to distributed sequence feature learning.

## Citation

If you use this repository for academic or research purposes, please cite the associated manuscript once published.

## License

This repository is made available for research and academic use.

