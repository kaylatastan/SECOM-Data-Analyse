
# 🏭 Semiconductor Manufacturing Defect Detection

**UCI SECOM Dataset - Machine Learning Pipeline**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Educational-green.svg)](https://archive.ics.uci.edu/ml/datasets/SECOM)

---

## 📋 Project Overview

This project implements a machine learning pipeline for **semiconductor manufacturing defect detection** using the UCI SECOM dataset. The goal is to predict pass/fail outcomes from 591 sensor measurements collected during the fabrication process.

### Key Features

- **Informative Missingness Strategy**: Engineered binary flags to capture sensor failures
- **Proper SMOTE Implementation**: Applied only inside CV folds to prevent data leakage
- **Multi-Model Comparison**: Random Forest, XGBoost, and Neural Network
- **Industry-Relevant Analysis**: Feature importance for process optimization

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SECOM) |
| Samples | 1,567 |
| Features | 591 sensor measurements |
| Target | Pass (1) / Fail (-1) |
| Class Imbalance | ~93.5% Pass, ~6.5% Fail |

### Data Usage & Licensing

The SECOM dataset is provided by the UCI Machine Learning Repository for **educational and research purposes**. When using this dataset:

- Cite: Dua, D. and Graff, C. (2019). UCI Machine Learning Repository
- This analysis is for educational demonstration only
- No confidential manufacturing data is included

---

## 🚀 Quick Start

### 1. Environment Setup

bash
# Clone repository
git clone <repo-url>
cd SECOM

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

```

### 2. Run the Pipeline

```bash
# Navigate to notebooks
cd notebooks

# Launch Jupyter
jupyter notebook 01_final_pipeline.ipynb

```

### 3. Execute Cells

Run all cells sequentially. Expected runtime: **10-15 minutes**

---

## 📁 Project Structure

```text
SECOM/
├── data/
│   └── uci-secom.csv           # Dataset
├── notebooks/
│   ├── 01_final_pipeline.ipynb       # Main pipeline (run this)
│   ├── 01_experiments_appendix.ipynb # PCA/KNN/SVM experiments
│   └── archive/                      # Original notebook backup
├── figures/                    # Generated plots
├── reports/                    # Model comparison CSVs
├── models/                     # Saved model files
├── requirements.txt
├── README.md
└── cv_bullets.md               # CV-ready bullet points

```

---

## 📈 Key Findings

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Random Forest | ~93% | Varies | Varies | Varies | ~0.80+ |
| XGBoost | ~93% | Varies | Varies | Varies | ~0.75+ |
| ANN | ~92% | Varies | Varies | Varies | ~0.55+ |

*Note: Exact metrics depend on run; see notebook outputs for precise values.*

### Feature Engineering Insights

* **Informative Missingness**: Created 100+ binary flags for columns with >5% missing data
* **Top Features**: Several missing indicator flags appeared in top 30 important features, validating the strategy
* **Sensor Patterns**: Missing data patterns correlate with defect outcomes

### Why PCA Was Not Used

PCA experiments (see Appendix notebook) showed:

1. No significant performance improvement
2. Loss of feature interpretability
3. Informative missingness flags lose meaning after transformation

---

## 🛠️ Technical Details

### Preprocessing

* **Imputation**: Median filling for missing values
* **Scaling**: StandardScaler normalization
* **Split**: Stratified 80/20 train/test (random_state=42)

### Class Imbalance Handling

* SMOTE applied **only inside CV folds** (prevents data leakage)
* Class weights used in tree-based models

### Reproducibility

* `random_state=42` set everywhere
* Model saving snippets included
* Full requirements.txt with pinned versions

---

## 📚 References

1. UCI SECOM Dataset: https://archive.ics.uci.edu/ml/datasets/SECOM
2. McCann, M., & Johnston, A. (2008). SECOM Dataset. UCI ML Repository.

---

## 📧 Contact

For questions about this project, please open an issue on the repository.

---

*Last updated: December 2025*

```

```

