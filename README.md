# Home Credit Default Risk

Predicting which loan applicants will have payment difficulties, based on the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset from Kaggle. The project covers the full workflow: exploratory analysis, feature engineering with five linked tables, model comparison, threshold analysis, feature importance and a fairness check.

## Results at a glance

- **Data:** 307,511 loan applications, of which 8.07% defaulted (about 11 repaid applicants per default). Because of this imbalance, accuracy would be misleading and the models are compared by AUC-ROC.
- **Best model:** tuned LightGBM with a validation AUC-ROC of **0.7796** (Logistic Regression baseline: 0.7606).
- **Model comparison:** LightGBM (0.7773), tuned LightGBM (0.7796) and XGBoost (0.7745) perform practically the same; the hyperparameter search brought no meaningful gain over the default settings.
- **From scores to decisions:** at a threshold of 0.5 the final model flags 28.9% of applicants and finds 67.8% of the defaults, with a precision of 19.0%. The threshold is a business decision that depends on the cost of a missed default compared to a rejected good applicant, which the dataset does not contain.
- **Drivers:** the three external scores (`EXT_SOURCE_1/2/3`) were already the strongest predictors in the EDA and carry about a third of the final model's gain. Seven of the model's top 15 features come from the linked tables.

## Notebooks

| Notebook | Content |
|---|---|
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory analysis of `application_train.csv`: data quality, target distribution, missing values (including the block of building features and the recurring "not currently employed" group), numerical and categorical features, correlations with the target |
| [`notebooks/02_feature_engineering.ipynb`](notebooks/02_feature_engineering.ipynb) | Data cleaning (placeholder values, income outlier), categorical encoding, handling of missing values, removal of redundant building features, and aggregated features from five linked tables (`bureau`, `previous_application`, `POS_CASH_balance`, `credit_card_balance`, `installments_payments`), including four custom ratio features. `bureau_balance` was left out on purpose. Result: 244 columns without missing values |
| [`notebooks/03_modeling.ipynb`](notebooks/03_modeling.ipynb) | Stratified train/validation split, Logistic Regression baseline, LightGBM (with hyperparameter search) and XGBoost, ROC and precision-recall curves, confusion matrix and threshold analysis, feature importance, fairness check across gender groups |

## Notes and limitations

- **Gender in the data:** the column `CODE_GENDER` contains only the values "M" and "F" (plus four rows with an unknown value, which were treated as missing). Other gender categories were not recorded, so the fairness check can only compare these two groups and says nothing about other gender identities.
- **Preprocessing and data leakage:** some preprocessing statistics (imputation medians, income cap, one-hot categories) were computed on the full training data before the train/validation split. This lets a small amount of validation information into the preprocessing; the scaling for the Logistic Regression was fitted on the training part only.
- **Evaluation:** all results come from a single validation split of the Kaggle training data. The models were not scored on the hidden Kaggle test set.
- **Threshold analysis:** the cost ratios used to illustrate threshold choices are assumptions, not values from the data.
- **Possible next steps:** features from `bureau_balance`, a wider hyperparameter search (two of the best values lay at the edge of the search range) and tuning of XGBoost.

## Tech stack

Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, matplotlib, seaborn. The notebooks were run on Kaggle.

## Data

The data is not included in this repository. It can be downloaded from the [competition page](https://www.kaggle.com/c/home-credit-default-risk/data).
