# ROGII Wellbore Geology Prediction

This repository contains the Python code I created for the ROGII Wellbore Geology Prediction Kaggle project. This workflow both horizontal and typewell data, prepares features, trains a regression model, and generates prediction

## Code

- 'horizontal_ingest.py' and 'horizontal_test_ingest.py' load horizontal well data.
- 'typewell_ingest.py' and 'typewell_test_ingest.py' load typewell data.
- 'horizontal_preprocess.py' and 'typewell_preprocess.py' prepare the data for modeling
- 'train_model.py' trains the model
- 'ingest.py' contains additional data loading code

## Data

The competition data is not included in this repository. The scripts expect that the relevant input files are locally available.
