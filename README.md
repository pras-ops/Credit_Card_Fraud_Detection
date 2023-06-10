# Credit Card Fraud Detection using PyCaret

This project focuses on detecting fraudulent credit card transactions using PyCaret, a low-code machine learning library in Python. The goal is to enable credit card companies to recognize and prevent fraudulent transactions, ensuring that customers are not charged for unauthorized purchases.

## Context

Credit card fraud detection is crucial for preventing financial losses and protecting customers. The dataset used in this project contains credit card transactions made by European cardholders in September 2013. It consists of 284,807 transactions, out of which 492 are classified as fraudulent, resulting in a highly unbalanced dataset where fraudulent transactions account for only 0.172% of the total.

The dataset consists of numerical input variables obtained through a PCA transformation, along with two non-transformed features: 'Time' and 'Amount'. The 'Time' feature represents the seconds elapsed between each transaction and the first transaction in the dataset, while the 'Amount' feature indicates the transaction amount. The response variable, 'Class', takes a value of 1 for fraudulent transactions and 0 for legitimate ones.

Due to confidentiality reasons, the original features and additional background information about the data are not provided.

To evaluate the model performance, the Area Under the Precision-Recall Curve (AUPRC) is recommended instead of the confusion matrix accuracy, given the class imbalance.

## Report

The following tasks were performed in this notebook:

1.  Data Analysis
2.  Feature Engineering
3.  Model Building and Prediction using ML Techniques
4.  Model Building and Prediction using PyCaret (Auto ML)

### Data Analysis

-   The dataset shape is (284,807, 31).
-   There are no null values in the dataset.
-   The distribution of normal and fraudulent cases in the dataset is highly imbalanced, with fraud cases accounting for a small fraction (0.172%) of the total.

### Feature Engineering

-   Independent features were created by selecting all columns except "Class".
-   The dependent variable, "Class", was assigned to the target variable, y.

### Model Building

-   The dataset was split into training and testing sets using a 70:30 ratio.
-   Two models were used for anomaly detection: Isolation Forest and OneClassSVM.

#### Isolation Forest

-   Isolation Forest is a technique based on isolating anomalies by randomly selecting features and split values.
-   The model was trained with 100 estimators and the same number of samples as the training data.
-   Anomaly predictions were made on the test set.
-   The predictions were mapped to 0 (normal) and 1 (fraud).
-   Accuracy score, classification report, and confusion matrix were computed.
-   The number of errors made by the Isolation Forest model was also calculated.

#### OneClassSVM

-   OneClassSVM is another anomaly detection algorithm based on the Support Vector Machine approach.
-   The model was trained using the radial basis function (rbf) kernel, with other hyperparameters set to default values.
-   Anomaly predictions were made on the test set.
-   The predictions were mapped to 0 (normal) and 1 (fraud).
-   Accuracy score, classification report, and confusion matrix were computed.
-   The number of errors made by the OneClassSVM model was also calculated.

### PyCaret (Auto ML)

-   PyCaret, a low-code machine learning library, was used to automate the machine learning pipeline.
-   The dataset was loaded into PyCaret.
-   A comparison of multiple classification models was performed using the `compare_models()` function.
-   The Random Forest algorithm was selected as the best-performing model based on the Kappa score.
-   Hyperparameter tuning was applied to the Random Forest model.
-   Predictions were made on the holdout dataset.

The final predictions and model evaluations are available in the notebook.

Please refer to the notebook for the complete code implementation and further details.
