# 🩺 Diabetes Prediction using Classification Algorithms

This project focuses on predicting whether a person is diabetic or not using medical data, with a special focus on dealing with **missing values**, **outliers**, **imbalanced data**, and **choosing the right classification model**.

---

## 📌 Objective

To predict the likelihood of diabetes using various features such as glucose level, BMI, age, blood pressure, etc., from the Pima Indians Diabetes Dataset. I explored multiple machine learning classification models and techniques to handle data quality issues and imbalances.

---

## 📊 Dataset Overview

The dataset contains information on female patients including:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age
- Outcome (0 = Non-Diabetic, 1 = Diabetic)

---

## 🧹 Data Cleaning & Imputation

I began by identifying missing values that were stored as **0s**, particularly in columns like Glucose, Insulin, BMI, etc.

- If the distribution was **symmetric** (no outliers), I used the **mean** for imputation.
- If the distribution was **skewed** (had outliers), I used the **median**.
- For **categorical features**, I would use the **mode**, though in this dataset all features were numerical.

I used **Seaborn’s displot** and **boxplots** to determine symmetry and outliers:
- `displot` helped visualize the shape of distributions.
- KDE lines indicated skewness and helped guide the imputation strategy.

```python
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())
...
```

---
## 📦 Outlier Detection and Removal

I applied Quantile-based filtering to remove outliers, particularly in the Insulin column which had high variance.

Instead of just removing rows from x_scaled, I made sure to apply the same mask to both features and labels to preserve alignment:

```python
q = x_scaled['Insulin'].quantile(0.95)
mask = x_scaled['Insulin'] < q
dataNew = x_scaled[mask]
y_outlier_detection = y_outlier_detection[mask]
```
---
## ⚖️ Handling Imbalanced Data

The dataset was imbalanced with more non-diabetic cases. I tried:

    Oversampling (to increase samples in minority class)

    Undersampling (to reduce samples from majority class)

Then I used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic examples for the minority class.

🔴 Note: SMOTE was applied before normalization to maintain true geometry of data.

---
## 🚀 Models Used

I applied and compared several classification models:
1. Logistic Regression

    Good baseline model

    Predicts probability of class using a sigmoid function

    Suitable since the target variable is binary (0/1)

2. Gaussian Naive Bayes

    Assumes features follow Gaussian distribution

    Calculates P(class | features) based on Bayes' Theorem

    Simple yet powerful probabilistic model

    Works well with continuous features like age, glucose, etc.

3. K-Nearest Neighbors (KNN)

    Non-parametric (doesn’t assume any distribution)

    Classifies a point based on majority of K neighbors

    Needs feature scaling

    Performance was good after handling outliers and imbalances

📈 KNN Model Performance (after SMOTE)

              precision    recall  f1-score   support

           0       0.85      0.67      0.75       159
           1       0.54      0.76      0.63        79

    accuracy                           0.70       238
   macro avg       0.69      0.72      0.69       238
weighted avg       0.75      0.70      0.71       238

    Class 0 (Non-Diabetic): High precision, moderate recall

    Class 1 (Diabetic): Lower precision, but better recall

    Good balance in F1-score

    Overall accuracy: 70%

---
## 🧠 Key Takeaways

- Preprocessing matters just as much as the model!
- Imputation method depends on distribution shape (mean vs median)
- Removing outliers carefully prevents misaligned labels
- SMOTE helps a lot with imbalanced datasets
- KNN and Naive Bayes performed reasonably well
- Logistic regression is still a good baseline model
