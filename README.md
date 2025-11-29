# Operational Insight: Bottleneck Detection

This project builds a cross-dataset bottleneck detection system for service operations using two event logs:

- **5k dataset:** labeled bottleneck events  
- **2k dataset:** unlabeled operational events  

The goal was to learn from the 5k ground-truth data and apply that logic to the 2k dataset, 
validating that predictions align with real operational issues such as duration spikes, SLA breaches, and rework patterns.

---

## Repository Structure

```
notebooks/
    01_data_exploration.ipynb
    02_feature_mapping.ipynb
    03_feature_engineering.ipynb
    04_model_training.ipynb
    05_business_validation.ipynb
    06_bottleneck_detector_2k.ipynb
    07_validate_business_logic_5k.ipynb

models/
    decision_tree_model.pkl
    random_forest_model.pkl
    best_model.pkl

data/
    raw/
    clean/
    processed/
outputs/
```

---

## Overview

**✔ Data Exploration** — understand structure, quality, and bottleneck behavior  
**✔ Feature Mapping** — align features between 5k (labeled) and 2k (unlabeled)  
**✔ Feature Engineering** — create a unified feature set for both datasets  
**✔ Model Training** — Decision Tree + Random Forest using shared features  
**✔ Business Validation** — confirm predictions reflect real operational friction  
**✔ Bottleneck Detection (2k)** — generate pseudolabels for unlabeled events  
**✔ Logic Verification (5k)** — compare predictions to true bottleneck labels  

---

## Key Result

Random Forest performed best and was used to label bottlenecks in the 2k dataset.  
Predictions aligned strongly with real operational issues, showing the model provides actionable insights.

---

## Tech Stack

Python · Pandas · NumPy · Scikit-Learn · Matplotlib · Seaborn · Joblib · Jupyter

---
