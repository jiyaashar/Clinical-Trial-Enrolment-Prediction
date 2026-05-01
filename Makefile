.PHONY: install load clean explore features model advanced insights export test all

# Install all dependencies
install:
	pip install pandas numpy scikit-learn matplotlib seaborn requests pytest

# Step 1: Load AACT data → raw_trials.csv (requires data/*.txt files)
load:
	python load_data.py

# Step 2: Clean data → cleaned_trials.csv
clean:
	python clean_data.py

# Step 3: EDA + clustering → plot1-plot8
explore:
	python explore_data.py

# Step 4: Feature engineering → model_ready.csv
features:
	python feature_engineering.py

# Step 5: Train 5 baseline models → plot9-plot13 + model_results.json
model:
	python train_models.py

# Step 6: Regression, neural network, hyperparameter tuning → plot14-plot16
advanced:
	python advanced_models.py

# Step 7: Risk scoring, therapeutic area analysis → plot17-plot19
insights:
	python insights.py

# Step 8: Export model for dashboard → model_export.json
export:
	python export_model.py

# Run tests
test:
	pytest tests/ -v

# Run full pipeline (requires data/*.txt files in data/ folder)
all: install load clean explore features model advanced insights export
