# College Football Prediction Model

## Overview
This project is a statistical model for predicting NCAA Division I football game outcomes.  
The model was trained on historical game data dating back to 1980 and attempts to estimate team strength through an elo-based system and predict game winners and spreads.

The goal of the project was to explore predictive modeling techniques and evaluate whether a custom model could outperform common media predictions.

---

## Dataset

- **25,000+ historical games** from 1980–2025
- Game outcomes, scores, team data, "excitement levels", stadium data
- Data preprocessing performed in Python through a CSV filtering program 

---

## Model Approach

The model estimates team strength over time using a combination of:

- Sequential parameter updates
- Batch learning methods
- Kalman filtering for time-varying team performance
- Feature engineering based on scoring and opponent strength

The model updates team ratings as new game results are observed.

---

## Results

During evaluation on the **2024–2025 season**:

- Correctly predicted game winners **73.1% of the time**

### Example: National Championship Prediction

| Source | Prediction |
|------|------|
| **This Model** | Under |
| ESPN | Over |
| Actual Result | Under |

(spread results to come)

---

## Tools Used

- Python
- pandas
- NumPy
- scikit-learn (for hyperparameter optimization)

---

