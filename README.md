# Predicting Clinical Trial Enrollment Success

## 1. Project Description

This project addresses one of the pharmaceutical industry's most expensive problems: clinical trial enrollment failure. Currently, 80% of clinical trials fail to meet their enrollment targets, costing the industry over $200 billion annually. Each delayed trial day costs sponsors $600K-$8M while patients wait longer for potentially life-saving treatments.
We will build a classification model to predict whether a clinical trial will successfully achieve at least 80% of its target enrollment within the planned timeline. By analyzing historical trial data from ClinicalTrials.gov, we'll identify which protocol characteristics (eligibility criteria complexity, number of sites, therapeutic area, study phase) are most predictive of enrollment success or failure.
This project addresses a real industry need: helping pharmaceutical sponsors make data-driven decisions about trial design and site selection before investing millions in doomed protocols.

## 2. Project Timeline

Weeks 1-2 (Feb 11-24): Data Collection & Exploration
- ClinicalTrials.gov API 
- Collect 5,000-10,000 completed Phase 2/3 trials
- Initial data exploration and summary statistics
- Create preliminary visualizations of enrollment patterns (clustering)


Week 3 (Feb 25-Mar 2): Data Cleaning
- Handle missing enrollment values
- Standardize date formats and validate ranges
- Remove outliers and duplicate trials
- Document data quality issues

Week 4 (Mar 3-9): Feature Engineering
- Parse eligibility criteria text to count inclusion/exclusion rules
- Extract protocol complexity metrics (number of arms, study duration)
- Encode categorical variables (therapeutic area, sponsor type)
- Create target variable: enrollment success (1/0)

Week 5 (Mar 16-22): First Check-In & Baseline Models
- Present preliminary visualizations
- Train baseline classification models (logistic regression, KNN)
- Show initial results
- Aligns with: Classification lectures (Intro to Classification, Decision Trees)

Week 6 (Mar 23-29): Advanced Model Training
- Train ensemble models (Random Forest from Decision Trees lecture)
- Model evaluation and comparison

Week 7 (Mar 30-Apr 5): Model Optimization
- Hyperparameter tuning with cross-validation
- Address class imbalance if needed

Week 8 (Apr 6-12): Regression Analysis & Feature Interpretation
- Add regression component: predict exact enrollment percentage (not just binary)
- Create feature importance visualizations
- Analyze which factors most impact enrollment

Week 9 (Apr 13-19): Advanced Techniques
- Apply logistic regression for probability estimates 
- Optimize model evaluation metrics

Week 10 (Apr 22-28): Second Check-In & Neural Networks
- Try to implement neural network model 
- Compare all model performances
- Build interactive dashboard for risk prediction

Week 11 (Apr 29-May 1): Final Report & Presentation
- Complete README documentation
- Polish GitHub repository
- Final submission May 1

## 3. Project Goals & Aims

Primary Goal: Predict whether a Phase 2 or Phase 3 clinical trial will achieve ≥80% of its target enrollment within the planned study duration, with ≥75% model accuracy.
1. Classification Model Performance (measurable via holdout test set):
- Achieve ≥75% accuracy on test set
- Achieve ≥70% precision (minimize false positives—incorrectly predicting success)

2. Feature Importance Analysis (measurable via feature ranking):
- Identify and rank the top 10 protocol characteristics that predict enrollment failure
- Quantify impact: "Each additional exclusion criterion reduces enrollment success probability by X%"
- Determine which therapeutic areas have highest vs. lowest success rates (using clustering techniques)

3. Risk Scoring System (measurable via stratified performance):
- Create a 0-100 enrollment risk score for new trial protocols using logistic regression
- Validate that trials scoring <30 have <40% historical success rate
- Validate that trials scoring >70 have >85% historical success rate

4. Multi-Model Comparison (measurable performance metrics):
- Compare at least 5 different approaches: Logistic Regression, KNN, Decision Trees, Random Forest, Neural Networks
- Demonstrate which model performs best for this specific problem
- Show performance trade-offs (accuracy vs. interpretability)

5. Actionable Insights (measurable via specific recommendations):
- Generate data-driven protocol optimization suggestions (Example: "Reducing exclusion criteria from 15 to 8 could improve enrollment probability by Y%")
- Identify optimal number of trial sites for different therapeutic areas

## 4. Data Collection 

Potential Data Sources:
1. Primary Source: ClinicalTrials.gov API v2
   URL: https://clinicaltrials.gov/api/v2/studies
   API Documentation: https://clinicaltrials.gov/data-api/api
   Legally required registration for trials conducted in the US
   Contains actual enrollment outcomes 
   Includes detailed protocol information (eligibility criteria, sites, phases, sponsors)
   Free access, no authentication required, unlimited requests

3. Secondary Sources:
   a. AACT Database (Clinical Trials Transformation Initiative)
      URL: https://aact.clinepi.org/
      Pre-processed relational database version of ClinicalTrials.gov
   b. FDA Drug Trials Snapshots
      URL: https://www.fda.gov/drugs/drug-approvals-and-databases/drug-trials-snapshots
      Demographic data for FDA-approved drug trials
      Can cross-reference to validate successful trial outcomes
      Useful for diversity analysis (secondary research question)

# Data Collection Method

Step 1: Data Collection
Using the ClinicalTrials.gov website for information about completed drug trials
Specifically: Phase 2 and Phase 3 trials that have finished and have real enrollment numbers
Extract details like: trial name, how many patients enrolled, where the trial happened, what the eligibility rules were, who sponsored it, etc.

Step 2: Getting the data in batches
Goal: collect between 5,000-10,000 trials total

Step 3: Basic quality checks while downloading
Skip trials that are missing important info (like enrollment numbers or dates)
Make sure the numbers make sense (no negative patients, no impossibly large numbers)
Keep track of any errors we encounter

Step 4: Saving the data
Original files: Save the raw data exactly as we receive it (as backup)
Cleaned version: Convert to a simple spreadsheet format (CSV file) that's easy to work with

What to expect:
To have about 5,000-10,000 trials worth of data

If the website is down or having issues, backup: download the entire database as a file from https://aact.clinepi.org/download

# Why This Project Matters
Every day a clinical trial is delayed costs sponsors $600K-$8M in lost revenue and delays patient access to potentially life-saving treatments. By predicting enrollment failures before trials start, sponsors can:
- Redesign overly restrictive eligibility criteria
- Allocate more sites to high-risk trials
- Choose better geographic regions for recruitment
- Save millions in sunk costs on doomed trial designs

This project demonstrates both technical data science skills (API integration, feature engineering, classification/regression modeling, neural networks) and pharmaceutical industry domain expertise—exactly what employers in pharma/biotech/healthcare are looking for.


