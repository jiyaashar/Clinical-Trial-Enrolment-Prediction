# Predicting Clinical Trial Enrollment Success


## How to Build and Run

### Prerequisites
- Python 3.9+
- AACT flat files (download "Flat Text Files" from https://aact.ctti-clinicaltrials.org/downloads, unzip into `data/` folder)

### Quick Start — Full Pipeline

```bash
git clone https://github.com/jiyaashar/Clinical-Trial-Enrolment-Prediction.git
cd Clinical-Trial-Enrolment-Prediction

# Download AACT data and unzip into data/ folder, then:
make all
```

`make all` runs every step: installs dependencies, loads data, cleans it, generates EDA plots, engineers features, trains all models, runs regression/neural network/tuning, generates insights, and exports the model for the dashboard.

### Step-by-Step

```bash
make install     # Install Python dependencies
make load        # Load AACT flat files → raw_trials.csv
make clean       # Clean data → cleaned_trials.csv
make explore     # EDA + K-Means clustering → plot1-plot8
make features    # Feature engineering → model_ready.csv
make model       # Train 5 baseline classifiers → plot9-plot13
make advanced    # Regression + neural network + tuning → plot14-plot16
make insights    # Risk scoring + therapeutic area analysis → plot17-plot19
make export      # Export trained model → model_export.json
```

### Run Tests

```bash
make test
```

23 tests covering age parsing, criteria counting, data file integrity, and target variable validation.

### How to Contribute

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch and open a Pull Request

### Environment

- **Python:** 3.9+ (tested with Anaconda Python 3.13)
- **OS:** macOS, Ubuntu, Windows
- **Dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, pytest (see `requirements.txt`)

---

## Project Description

Clinical trial enrollment failure is one of the pharmaceutical industry's most expensive problems — 80% of trials fail to meet their enrollment targets, costing over $200 billion annually. Each delayed trial day costs sponsors $600K–$8M while patients wait longer for potentially life-saving treatments.

### What This Project Actually Predicts

This project does **not** predict whether a drug will work or whether a trial will produce positive clinical results (that depends on biology, chemistry, and factors no model can forecast from protocol data alone).

Instead, it predicts the operational aspect: **will this trial recruit enough patients?** Specifically, if it will achieve at least 80% of the enrollment expected for a trial of its type?

While on the surface, yes — enrollment = sites × patients_per_site. But the real question isn't arithmetic. The real question is: **before a trial starts, given only what you know about the protocol design, can you predict whether it will struggle to recruit?**

The challenge is that `enrollment_per_site` isn't known in advance but is the outcome. A trial designer decides the number of sites, the eligibility criteria, the phase, the therapeutic area, and the study duration. They do NOT know how many patients each site will actually recruit. That depends on dozens of factors: how restrictive the eligibility criteria are (more exclusion criteria = fewer eligible patients), what disease is being studied (schizophrenia trials struggle at 52% success vs. HIV at 65%), how many competing trials are running at the same sites, the local patient population, the site's recruitment experience, and so on.

This model learns the complex interaction between all of these protocol-level decisions and predicts the likely outcome. It tells a trial designer: "based on 9,517 historical trials with similar protocol characteristics, here's how likely your trial is to meet enrollment targets." That's something a simple multiplication can't do.

### Why This Is Useful

Trial sponsors typically invest $10M–$50M before the first patient is enrolled. If the protocol design is likely to cause enrollment failure, it's far cheaper to adjust the design (add sites, simplify eligibility, adjust the age range) BEFORE launching than to discover the problem 18 months in. This model provides that early warning signal.

**Important caveat:** The model's top predictor, `enrollment_per_site`, is derived from actual enrollment data, which creates a data leakage concern. In a production deployment, this feature would be estimated from the historical performance of the specific sites being considered, rather than from the trial's own outcome. The model still works without this feature (the other 25 features achieve ~82% accuracy via Logistic Regression), but it's most powerful when combined with site-level historical estimates.

**Primary Goal:** Predict enrollment success/failure with ≥75% accuracy and ≥70% precision.

**Result:** Tuned Decision Tree achieved **96.9% accuracy** and **94.8% precision** — exceeding both targets.

---

## Project Structure

```
├── .github/workflows/
│   └── tests.yml                  # CI: runs tests on push/PR
├── data/                          # AACT .txt files (gitignored, ~2 GB)
├── tests/
│   └── test_pipeline.py           # 23 unit tests
├── load_data.py                   # Step 1: Load AACT flat files → raw_trials.csv
├── clean_data.py                  # Step 2: Clean data → cleaned_trials.csv
├── explore_data.py                # Step 3: EDA + K-Means clustering → plot1-plot8
├── feature_engineering.py         # Step 4: Engineer 26 features → model_ready.csv
├── train_models.py                # Step 5: 5 baseline classifiers → plot9-plot13
├── advanced_models.py             # Step 6: Regression + neural net + tuning → plot14-plot16
├── insights.py                    # Step 7: Risk scoring + recommendations → plot17-plot19
├── export_model.py                # Step 8: Export model for dashboard → model_export.json
├── dashboard.html                 # Interactive enrollment risk predictor
├── raw_trials.csv                 # 10,000 raw trials
├── cleaned_trials.csv             # 9,517 cleaned trials
├── model_ready.csv                # 9,517 trials with 26 features + target
├── model_results.json             # Baseline model metrics
├── advanced_results.json          # Advanced model metrics
├── insights_results.json          # Risk scoring + therapeutic area results
├── model_export.json              # Exported Decision Tree for dashboard
├── plot1-plot19                    # All visualizations
├── Makefile                       # Build commands
├── requirements.txt               # Python dependencies
└── README.md                      # This file (final report)
```

---

## Data Collection

**Source:** [AACT Database](https://aact.ctti-clinicaltrials.org/downloads) — a publicly available relational database maintained by the Clinical Trials Transformation Initiative, mirroring all data from ClinicalTrials.gov. (contains files like facilities, ids, descriptions, designs, etc).

**Method:** Downloaded the "Flat Text Files" export (~2.2 GB). Script `load_data.py` reads 7 tables (studies, eligibilities, facilities, sponsors, design_groups, interventions, conditions), filters them, and merges by trial ID (`nct_id`).

**Filtering pipeline:**

| Step | Filter | Remaining |
|------|--------|-----------|
| Start | All studies in AACT | 578,106 |
| 1 | Completed trials only | 315,589 |
| 2 | Interventional (drug/device) | 247,034 |
| 3 | Phase 2, Phase 3, or Phase 2/3 | 62,658 |
| 4 | Actual enrollment reported and > 0 | 55,446 |
| 5 | Random sample for development | 10,000 |

We filter to completed Phase 2/3 interventional trials because these are the large scale trials where enrollment failure actually costs millions. Phase 1 trials are tiny safety studies (10-30 people) where enrollment is rarely a problem.

---

## Data Cleaning

Implemented in `clean_data.py`. The script produces a data quality report before and after cleaning.

| Issue | Count | Action | Reasoning |
|-------|-------|--------|-----------|
| Duplicate trial IDs | 0 | N/A | Already deduplicated |
| Unparseable dates | 261 | Dropped | Need both dates for duration calculation |
| Unreasonable duration (<7d or >20yr) | 26 | Dropped | Likely data entry errors |
| Enrollment outliers (<1st or >99th pctile) | 196 | Dropped | Extreme values distort models |
| `why_stopped` 100% missing | All | Ignored column | Expected - all trials are completed |
| `maximum_age` 49% missing | 4,945 | Filled with median (65) in feature engineering | Many trials don't set upper age limit |

Categorical fields (phase, sponsor class, sex) were uppercased and stripped of whitespace for consistency.

**Result:** 10,000 → 9,517 trials (483 removed, 4.8% loss).

---

## Feature Extraction

Implemented in `feature_engineering.py`. We created 26 numeric features from the raw trial data.

| Feature | Description | How Extracted |
|---------|-------------|---------------|
| `duration_days` | Study length in days | Computed from start/completion dates |
| `num_sites` | Number of trial locations | Counted from facilities table |
| `num_countries` | Countries involved | Distinct country count from facilities |
| `num_arms` | Treatment groups | From design_groups table |
| `num_interventions` | Drugs/devices tested | From interventions table |
| `enrollment_per_site` | Patients per site | enrollment_actual / num_sites |
| `has_multiple_arms` | Binary: >1 arm? | Derived from num_arms |
| `num_inclusion_criteria` | Inclusion rule count | Parsed from eligibility text using `~*` delimiters |
| `num_exclusion_criteria` | Exclusion rule count | Parsed from eligibility text |
| `total_criteria` | Total eligibility rules | inclusion + exclusion |
| `criteria_text_length` | Eligibility text length | Proxy for protocol complexity |
| `min_age_years` | Minimum age (years) | Parsed from strings like "18 Years" |
| `max_age_years` | Maximum age (years) | Parsed; missing filled with median (65) |
| `age_range_years` | Eligible age range | max_age - min_age |
| `sex_restricted` | Binary: one sex only? | 1 if not "ALL" |
| `is_phase2`, `is_phase3` | Phase indicators | From phase string |
| `is_industry_sponsored` | Binary: industry sponsor? | From sponsor class |
| `is_cancer` through `is_schizophrenia` | 8 therapeutic area flags | Keyword match in conditions text |

**Target Variable:** Since ClinicalTrials.gov doesn't provide separate planned vs actual enrollment for completed trials, we compare each trial's enrollment to the median for its phase + sponsor group. Success = reached ≥80% of group median. Result: 59% success rate (5,615 successes, 3,902 failures) — balanced classes.

---

## Model Training & Evaluation

### Baseline Models (`train_models.py`)

All models evaluated with 80/20 stratified train/test split + 5-fold cross-validation.

| Model | Accuracy | Precision | Recall | F1 | CV Accuracy |
|-------|----------|-----------|--------|----|-------------|
| Decision Tree | 95.0% | 94.8% | 96.9% | 0.958 | 94.6% ± 0.8% |
| Random Forest | 90.3% | 90.3% | 93.6% | 0.919 | 90.6% ± 0.9% |
| Logistic Regression | 81.8% | 85.5% | 83.3% | 0.844 | 80.3% ± 0.6% |
| KNN (k=7) | 71.3% | 76.1% | 74.9% | 0.755 | 69.2% ± 1.3% |
| Naive Bayes | 64.1% | 85.4% | 47.3% | 0.609 | 63.7% ± 0.9% |

### Advanced Models (`advanced_models.py`)

**Regression** predicting log(enrollment) as a continuous value:

| Model | R² | RMSE |
|-------|----|------|
| Ridge Regression | 0.560 | 0.812 |
| Linear Regression | 0.560 | 0.812 |
| Lasso Regression | 0.527 | 0.842 |

R² = 0.56 means protocol features explain 56% of enrollment variance. The remaining 44% depends on unmeasured factors like recruitment budget and site experience.

**Neural Network** MLP classifier with ReLU activation and Adam optimizer:

| Architecture | Accuracy | F1 |
|-------------|----------|----|
| MLP (64 neurons) | 89.9% | 0.916 |
| MLP (128, 64) | 87.9% | 0.898 |
| MLP (128, 64, 32) | 86.9% | 0.890 |

Simplest architecture performed best as tabular data doesn't benefit from deep networks.

**Hyperparameter Tuning** — GridSearchCV with 5-fold CV:

| Model | Before Tuning (F1) | After Tuning (F1) | Best Parameters |
|-------|--------------------|--------------------|-----------------|
| Decision Tree | 0.958 | **0.974** | max_depth=15, min_samples_leaf=5 |
| Random Forest | 0.919 | 0.952 | max_depth=None, n_estimators=100 |

### Risk Scoring System (`insights.py`)

Logistic Regression probabilities scaled to 0-100:

| Score Range | Trials | Actual Success Rate | Target | Status |
|-------------|--------|---------------------|--------|--------|
| ≤ 30 (high risk) | 2,250 | 11.9% | < 40% | ✓ PASS |
| 31-69 (moderate) | 3,325 | 50.4% | — | — |
| ≥ 70 (low risk) | 3,942 | 93.2% | > 85% | ✓ PASS |

### Feature Importance

Both Random Forest importance and Logistic Regression coefficients agree on the top predictors:

| Rank | Feature | RF Importance | LR Coefficient | Impact per Unit |
|------|---------|---------------|-----------------|-----------------|
| 1 | enrollment_per_site | 35.0% | +6.07 | +0.61 pp |
| 2 | num_sites | 20.4% | +4.32 | +2.06 pp |
| 3 | duration_days | 5.2% | +0.18 | +0.004 pp |
| 4 | num_arms | 4.5% | +0.18 | +3.51 pp |
| 5 | has_multiple_arms | 3.8% | +0.38 | +20.8 pp |

Factors that hurt enrollment (LR coefficients): is_industry_sponsored (−0.55), is_phase3 (−0.48), num_countries (−0.31).

### Actionable Recommendations

1. **Add more sites** - each additional site improves success probability by 2.06 percentage points. Failed trials have median 1 site vs 4 for successes.
2. **Simplify eligibility** - reducing exclusion criteria from 15 to 8 improves probability by ~0.8 percentage points.
3. **Plan adequate duration** - successful trials run median 915 days vs 793 for failures.
4. **Therapeutic area matters** - HIV (65.3%) and Cancer (64.6%) have highest success rates. Schizophrenia (51.9%) is hardest.
5. **Optimal site count varies** - Cancer/Diabetes trials need 100+ sites. Depression peaks at 2-5 sites.

---

## Limitations

1. **Target variable is a proxy** - we don't have true planned enrollment, so we compare to group median. With real target enrollment data, accuracy would likely be lower.
2. **enrollment_per_site has data leakage risk** - it's derived from actual enrollment (the outcome). In production, this would be estimated from historical site performance.
3. **Regression R² = 0.56** — protocol features alone can't fully predict enrollment. Recruitment budget, competing trials, and site experience are unmeasured factors.
4. **Sample size** - we used 10,000 of 55,000 available trials for development speed.
5. **Eligibility criteria parsing** - counting `~*` delimiters is heuristic; some criteria may be miscounted.

---

## Data Visualizations

### EDA Plots (`explore_data.py`)

| Plot | What It Shows |
|------|--------------|
| `plot1` | Enrollment distribution — heavily right-skewed, median 103 |
| `plot2` | Enrollment by phase - Phase 3 enrolls ~4× more than Phase 2 |
| `plot3` | Enrollment by sponsor - industry trials have highest median (175) |
| `plot4` | Sites vs enrollment - clear positive log-log relationship |
| `plot5` | Top 15 conditions - breast cancer and diabetes most studied |
| `plot6` | Correlation heatmap - num_sites and num_countries correlated at 0.64 |
| `plot7` | K-Means clustering (k=3) - natural groupings of small/medium/large trials |
| `plot8` | Elbow method - k=3 is reasonable, no sharp elbow |

### Model Plots (`train_models.py`)

| Plot | What It Shows |
|------|--------------|
| `plot9` | 5 model comparison across accuracy, precision, recall, F1 |
| `plot10` | Confusion matrices - Decision Tree has only 95 errors out of 1,904 |
| `plot11` | Random Forest feature importance - enrollment_per_site (35%) and num_sites (20%) dominate |
| `plot12` | Logistic Regression coefficients - green (helps) vs red (hurts enrollment) |
| `plot13` | Cross-validation accuracy with error bars confirms no overfitting |

### Advanced Plots (`advanced_models.py`)

| Plot | What It Shows |
|------|--------------|
| `plot14` | Regression predicted vs actual + residual plot (R² = 0.56) |
| `plot15` | Neural network confusion matrices - 3 architectures compared |
| `plot16` | All 8 models ranked by F1 - tuned Decision Tree wins at 0.974 |

### Insights Plots (`insights.py`)

| Plot | What It Shows |
|------|--------------|
| `plot17` | Therapeutic area success rates colored by K-Means clustering |
| `plot18` | Risk score validation - histogram + calibration curve |
| `plot19` | Success rate by number of sites - more sites = better |

### Interactive Dashboard 
An interactive dashboard that toggles with numbers of major features and gives an estimate of what the prediction could be was created that can be found as "dashboard.html". 

However, it does have limitations pertaining to the predicitions and outcomes.
The current implementation uses a shallower Decision Tree (depth 5) for smoother probability estimates, which sacrifices some accuracy compared to the full tuned model (depth 15). A better approach would be to export a Random Forest or Logistic Regression model instead, as these naturally produce smoother probability distributions across the feature space. The dashboard would also benefit from showing confidence intervals around its predictions rather than a single point estimate, and from incorporating real-time data on currently recruiting competing trials to adjust the enrollment difficulty dynamically.


### Future Directions
While the current model achieves strong classification performance (97.4% F1), several improvements would make it more practically useful.
The most significant issue is the target variable proxy. We compare each trial's enrollment to its phase/sponsor group median because the AACT database doesn't separate planned vs actual enrollment for completed trials. Accessing the original planned enrollment numbers either through the ClinicalTrials.gov protocol amendments or through industry partnerships would give us a true target and likely produce more nuanced predictions. Related to this, enrollment_per_site is currently derived from actual enrollment, creating a circularity where the model partially predicts the outcome from itself. In a production deployment, this feature would need to be replaced with historical site-level recruitment rates from previous trials at those same facilities, which organizations like Medidata and IQVIA track but don't make publicly available.

The model could also benefit from additional features not available in the current dataset: the recruitment budget allocated per site, the number of competing trials recruiting similar patients at the same time, the geographic density of eligible patients near each site, the experience level of site investigators, and the specific drug mechanism (which affects patient willingness to participate). Natural language processing on the full protocol text beyond our simple criteria counting could capture protocol complexity more richly.

Finally, the model was trained on all completed Phase 2/3 trials regardless of when they ran. A time-aware model that accounts for shifts in enrollment patterns over the last two decades such as the increasing difficulty of recruitment due to trial proliferation and the impact of decentralized trial designs post-COVID would likely generalize better to trials starting today.
