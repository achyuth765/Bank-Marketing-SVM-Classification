# Bank Marketing – SVM Classification

## Project Overview

This project applies **Support Vector Machine (SVM)** classification to the Bank Marketing dataset to predict whether a client will subscribe to a term deposit.

The project investigates different SVM kernels, hyperparameter tuning, support vectors, and class-imbalance handling.

## Dataset

The project uses the **Bank Marketing** dataset (`bank-additional-full.csv`).

The target variable is:

- `yes` – Client subscribes to a term deposit
- `no` – Client does not subscribe

The dataset is highly imbalanced, with the `no` class representing the majority of observations.

## Data Preprocessing

The following preprocessing steps are performed:

- The `duration` feature is removed because it represents information only known after the call and can cause target leakage.
- `unknown` categorical values are retained as an explicit category.
- A `contacted_before` feature is created from `pdays`.
- `pdays` is transformed into `pdays_adj`, replacing the `999` sentinel value with `-1`.
- Categorical variables are **One-Hot Encoded**.
- Numerical variables are **Standardised** using `StandardScaler`.
- A **70:30 stratified train/test split** is used.
- A training subsample of 4,000 observations is used for SVM fitting to control computational cost.

## Why SVM?

SVM is a margin-based classification algorithm.

Since SVM depends on distances and margins:

- Numerical features need to be standardised.
- Categorical features need to be encoded.
- Kernel functions can be used to model non-linear decision boundaries.

The analysis reports **accuracy, balanced accuracy, precision, recall, F1-score, and ROC-AUC** because accuracy alone can be misleading for this imbalanced dataset.

## Q1 – Linear Kernel SVM

A Linear SVM with the default:

```text
C = 1.0
```

is trained and evaluated on the held-out test set.

The decision function is also used to calculate ROC-AUC.

## Q2 – Linear SVM C Sweep

Different values of `C` are tested:

```text
0.001
0.01
0.1
1
10
100
```

Three-fold stratified cross-validation is used with **F1-score** as the selection metric.

The experiment also examines how changing `C` affects:

- Model performance
- Margin width
- Number of support vectors

A smaller `C` gives stronger regularisation and generally allows a wider margin, while a larger `C` places a higher penalty on training errors.

## Q3 – RBF and Polynomial Kernel Tuning

Two non-linear SVM kernels are investigated.

### RBF Kernel

The following parameters are tuned:

- `C`
- `gamma`

The best RBF configuration is:

```text
C = 10
gamma = scale
```

It achieves a test accuracy of approximately **89.97%** and a test F1-score of approximately **0.404**.

### Polynomial Kernel

The polynomial kernel is tuned using:

- `C`
- `degree`
- `gamma`

The best configuration is:

```text
C = 1
degree = 4
gamma = scale
```

Its test accuracy is approximately **89.23%**, with a test F1-score of approximately **0.365**.

The tuned RBF model therefore performs better than the tuned polynomial model based on the reported F1-score.

## Q4 – Support Vectors

The best-performing kernel model is further examined through its support vectors.

Support vectors are the training observations closest to, or violating, the SVM decision margin.

A high number of support vectors indicates substantial overlap between the `yes` and `no` classes, making the classification problem more difficult.

The analysis also considers the computational implications of a large support-vector set.

## Q5 – Class Imbalance Handling

Because the target classes are highly imbalanced, **`class_weight='balanced'`** is used.

This approach increases the penalty associated with minority-class errors without changing the original training distribution.

The balanced RBF configuration uses:

```text
C = 10
gamma = 0.01
```

It achieves:

- Test accuracy: **84.13%**
- Balanced accuracy: **75.05%**
- Precision: **37.81%**
- Recall: **63.31%**
- F1-score: **47.35%**
- ROC-AUC: **78.90%**

The higher recall shows that class weighting substantially improves detection of the minority `yes` class, although overall accuracy decreases.

## Key Findings

- The dataset has significant class imbalance, making accuracy alone an unreliable performance measure.
- Linear SVM performance changes as the regularisation parameter `C` is varied.
- The RBF kernel performs better than the polynomial kernel based on test F1-score.
- Increasing `C` too much can significantly reduce model performance; very large values such as `C=100` and `C=1000` show much lower test F1-scores.
- Class weighting improves minority-class recall and F1-score.
- The balanced RBF model provides a better balance between identifying potential subscribers and maintaining reasonable precision.

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (for optional SMOTE comparison)

## Project Structure

```text
bank-marketing-svm/
│
├── Untitled1(2).ipynb
├── bank-additional-full.csv
├── data.joblib
├── q1_linear_default.joblib
├── q2_C_sweep.joblib
├── q2_summary.json
├── q3_kernel_tuning.joblib
├── q3_summary.json
├── q4_sv_inspection.joblib
├── q5_balanced.joblib
├── q5_summary.json
└── README.md
```

## How to Run

1. Clone or download this repository.
2. Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

3. Open:

```text
Untitled1(2).ipynb
```

using Jupyter Notebook or JupyterLab.

4. Keep `bank-additional-full.csv` in the same project directory.
5. Run the notebook cells from beginning to end.

## Conclusion

This project demonstrates the use of **SVM for binary classification on an imbalanced banking dataset**.

Different kernels and hyperparameters are evaluated, followed by an analysis of support vectors and class-imbalance strategies.

The results show that **RBF SVM with class weighting** provides improved minority-class detection, making it a useful configuration when identifying potential term-deposit subscribers is more important than maximising raw accuracy.
