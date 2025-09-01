

# Credit Card Fraud Detection

This project applies machine learning to detect fraudulent credit card transactions using a custom dataset containing transaction details, merchant information, and fraud labels. The workflow includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

##  Features

* Loads and preprocesses the dataset from `credit_card_fraud_dataset.csv`
* Converts categorical features like `TransactionType` and `Location` into numerical form using encoding
* Extracts date-time components (day, month, hour, weekday) from `TransactionDate`
* Handles imbalanced data using oversampling (SMOTE) or undersampling techniques
* Trains multiple models such as:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Evaluates model performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * ROC-AUC

##  Dataset

**File:** `credit_card_fraud_dataset.csv`
**Size:** 100,000 transactions
**Download Link:** [Download Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

**Columns:**

1. `TransactionID` – Unique transaction identifier
2. `TransactionDate` – Timestamp of the transaction
3. `Amount` – Transaction amount (USD)
4. `MerchantID` – Unique merchant identifier
5. `TransactionType` – Purchase or refund
6. `Location` – City where the transaction took place
7. `IsFraud` – Target variable (1 = Fraud, 0 = Legitimate)

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/akaaash04/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

##  Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
imbalanced-learn
jupyter
```

## ▶ Usage

Run the Jupyter Notebook to execute the full pipeline:

```bash
jupyter notebook credit_card.ipynb
```

Or, convert and run as a Python script:

```bash
python credit_card.py
```

## � Model Workflow

1. **Data Loading** – Read CSV file using Pandas
2. **Preprocessing**

   * Convert `TransactionType` and `Location` to numeric codes
   * Extract date-time features from `TransactionDate`
   * Normalize `Amount`
3. **Handling Class Imbalance** – Apply SMOTE or undersampling
4. **Model Training** – Fit multiple ML models and tune hyperparameters
5. **Evaluation** – Compare models using classification metrics and ROC curves
6. **Visualization** – Confusion matrices, feature importance plots


