# 🩺 Diabetes Prediction using Machine Learning

A classification project to predict the onset of diabetes based on diagnostic measurements (Pima Indians Diabetes Dataset).

## 📌 Project Overview
- Dataset: Pima Indians Diabetes Database (768 records, 9 features)
- Objective: Binary classification — predicting whether a patient has diabetes (Outcome: 1) or not (Outcome: 0)

## 🛠️ Tech Stack & Libraries
- Language: Python
- Data Manipulation: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- Machine Learning: Scikit-Learn (Logistic Regression, Decision Tree Classifier)

## 📊 Workflow
1. Exploratory Data Analysis (EDA): info(), isna().sum(), describe(), and value_counts() for class balance checks
2. Data Visualization: Class distribution with countplot and feature correlation analysis with a Seaborn heatmap
3. Data Preprocessing: Train-test split (test_size=0.2, random_state=42)
4. Model Training & Comparison: Training and evaluating Logistic Regression and Decision Tree models in a loop
5. Inference: Predicting diabetes for a custom patient profile

## 📈 Results
- Logistic Regression: ~74.67% Accuracy
- Decision Tree Classifier: ~75.97% Accuracy

## 🚀 How to Run
1. Clone the repository
2. Open diabet.ipynb in VS Code or Jupyter Notebook
3. Run all cells (Run All)

---
⭐ Part of my [Python Data Analysis Portfolio](https://github.com/alipyaiengineer2010-ai/python-data-analysis-portfolio)
