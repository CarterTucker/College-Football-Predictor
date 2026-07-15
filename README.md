# College Football Prediction Model

## Overview
This project is a statistical model for predicting NCAA Division I football game outcomes. It evolved through several versions (v0–v5), moving from a simple margin-based rating system to a Kalman-filtered model with automated hyperparameter tuning. The current model estimates team strength through a variance-aware rating system and predicts game winners and spreads.

The goal of the project was to explore predictive modeling techniques and evaluate whether a custom model could outperform common media predictions.

---

## Dataset

- **24,900+ games** spanning the 1990–2026 seasons (a season's bowl games and championship roll into the following January, hence the 2026)
- FBS-vs-FBS games only, pulled from the [CollegeFootballData.com](https://collegefootballdata.com) API
- Fields: date, season, season type, neutral-site flag, conference-game flag, venue, home/away teams and scores, and a computed scoring margin
- Data collection and filtering handled by `conditioner.py`

---

## Model Evolution

Each version added a layer of sophistication over the last:

### v0 — Baseline Rating (`projector_v0.py`)
A simple Elo-style system with a single fixed learning rate (`β = 0.03`). Team ratings update in proportion to the error between the actual scoring margin and the margin implied by both teams' current ratings. No home-field advantage, no season-to-season adjustment.

### v1 — Home Field & Season Decay (`projector_v1.py`)
Adds a fixed home-field advantage constant (`HFA ≈ 4.9`) and a season-boundary rule: when the gap between two consecutive games exceeds 150 days (the offseason), every team's rating decays by 20% to account for roster turnover. Learning rate increased to `β = 0.095`.

### v2 — Kalman-Filtered Ratings (`projector_v2.py`, `cfb_predictor_lib.py`)
Shared logic — loading games, initializing teams, detecting season changes — is pulled into a reusable library, `cfb_predictor_lib.py`. The rating update itself becomes a Kalman filter: each team carries a strength *and* a variance (uncertainty), and the update weight (`K_var`) is derived from that uncertainty instead of a fixed learning rate. Home-field advantage is now learned via gradient descent rather than hand-set. Evaluation is restricted to games after 2010 to let ratings stabilize first.

### v3/v4 — Train/Test Framework & Automated Tuning (`cfbModelFramework.py`, `experiments.py`)
`cfbModelFramework.py` extends the shared library with train/test awareness — `loadGames` can filter to a specific year range, and `initializeTeams` either trains fresh ratings or carries a team's previously trained strength/variance into a test period, with a cold-start fallback for teams with no training history. `experiments.py` uses this framework together with `scipy.optimize.minimize` (Nelder-Mead) to automatically search for the best values of the model's four hyperparameters (`gamma`, `R_var`, `Q_week`, `Q_offseason`) — training on one span of seasons and minimizing error on a later one, replacing manual trial-and-error tuning.

### v5 — Final Model (`___`)
*(fill in — this is the version the results below are drawn from)*

---

## Results

Results below reflect the final model (v5), evaluated on the **2024–2025 season**:

- Correctly predicted game winners **73.1% of the time**

### Example: National Championship Prediction

| Source | Prediction |
|------|------|
| **This Model** | Under |
| ESPN | Over |
| Actual Result | Under |

**Predicted Spread:** `___` *(fill in)*

---

## Repository Structure

```
.
├── README.md
├── games.csv               # collected game data (1990–2026 seasons)
├── conditioner.py           # pulls & filters data from the CFBD API into games.csv
├── projector_v0.py          # v0 — baseline rating system
├── projector_v1.py          # v1 — adds home-field advantage + season decay
├── projector_v2.py          # v2 — Kalman-filtered ratings
├── cfb_predictor_lib.py     # shared helper functions used by v2
├── cfbModelFramework.py     # v3/v4 — train/test-aware framework
├── cfbModel.py               # train()/test() logic used by experiments.py
├── experiments.py           # v3/v4 — automated hyperparameter search
└── ___                       # v5 — final model (fill in filename)
```

---

## Tools Used

- Python
- pandas
- SciPy (`scipy.optimize.minimize`, Nelder-Mead) for hyperparameter search
- ___ *(anything else v5 relies on that isn't listed above — e.g. NumPy, scikit-learn)*
