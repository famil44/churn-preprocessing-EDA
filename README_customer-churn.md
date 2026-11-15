# Customer Churn: EDA & Preprocessing

Exploratory data analysis and preprocessing of a customer churn dataset, preparing it for churn prediction modelling.

## Overview

This notebook explores a customer churn dataset to understand who the customers are and how the target (`Churn`) is distributed, then encodes the categorical features so the data can be used by machine learning models.

## Dataset

- **File:** `Customer-Churn.csv`
- **Target:** `Churn` (Yes / No)
- **Key columns used:** `MonthlyCharges`, `Contract`, `InternetService`, `PaymentMethod`, `Churn`
- **Source:** *add the link to where you got the dataset (e.g. Kaggle) here*

## What the notebook does

### 1. Data inspection
- Preview the data with `head()` and `tail()`
- Check structure and data types with `info()`
- Summary statistics for numeric and categorical columns with `describe()`
- Check for missing values (`isnull().sum()`) and duplicate rows (`duplicated().sum()`)
- Explore the categories in `Contract` and their counts

### 2. Exploratory data analysis
- Histogram with KDE of `MonthlyCharges` to see its distribution
- Boxplot of `MonthlyCharges` to spot outliers
- Count plots for `Contract` and `Churn` to see category sizes and class balance
- Correlation matrix and heatmap of the numeric columns

### 3. Outlier detection
- IQR method on `MonthlyCharges`: bounds are `Q1 - 1.5 * IQR` and `Q3 + 1.5 * IQR`
- Count of rows falling outside those bounds

### 4. Encoding categorical variables
- **Label encoding:** `Churn` mapped to `No = 0`, `Yes = 1`
- **One-hot encoding** with `pd.get_dummies(drop_first=True)` on `Contract`, `InternetService` and `PaymentMethod`
- Boolean dummy columns converted to integers (`0` / `1`)

## Tech stack

- Python 3
- pandas
- NumPy
- matplotlib
- seaborn

## How to run

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Upload `Customer-Churn.csv` to your environment.
3. Update the file path in the first cell if needed (the notebook uses `/content/Customer-Churn.csv`, the Colab default).
4. Run all cells (**Runtime → Run all** in Colab).

## Next steps

- [ ] Include `Churn` in the correlation analysis (it is encoded after the heatmap is drawn)
- [ ] Encode the remaining text columns and check for hidden missing values (e.g. numeric columns stored as text)
- [ ] Split into train and test sets
- [ ] Train and compare models (Logistic Regression, Random Forest, XGBoost)
- [ ] Evaluate with precision, recall, F1, ROC-AUC and a confusion matrix
- [ ] Add feature importance and business recommendations to reduce churn

## Author

*Your name* · [LinkedIn](https://www.linkedin.com/) · [GitHub](https://github.com/)
