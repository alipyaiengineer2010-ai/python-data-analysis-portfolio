# 💼 Employee Salary Prediction with XGBoost

## 📌 Project Overview
Predicting employee salaries from demographic and job features using XGBoost (Extreme Gradient Boosting) — achieving an R² score of 99% — plus model interpretability through feature importance analysis.

## 📊 Dataset
- Source: Employers_data.csv (10,000 employee records)
- Features: Age, Gender, Department, Job Title, Experience Years, Education Level, Location
- Target: Salary
- Cleaning: Dropped non-informative columns (Employee_ID, Name, Location); verified zero missing values with isna().sum().
- Note: The dataset is synthetic, which explains the very high R² score.

## 🛠 Tech Stack
- Language: Python
- Libraries: Pandas, Scikit-Learn, XGBoost
- Model: XGBRegressor
- Encoding: One-Hot Encoding (pd.get_dummies)

## 🔄 Workflow
1. EDA: head(), info(), missing-value check (10,000 non-null across all columns).
2. Cleaning: Removing identifier and irrelevant text columns.
3. Encoding: One-Hot Encoding categorical features with dtype=int.
4. Split: 80/20 train-test split (random_state=42).
5. Training: XGBoost Regressor.
6. Evaluation: R² Score ≈ 0.991.
7. Interpretability: xgb.plot_importance to visualize which features drive salary.

## 📈 Results
- R² Score: 99.1%
- Top salary drivers: Age and Experience_Years dominate, followed by Gender and Job_Title_Engineer — consistent with real-world salary intuition.

## 💡 Key Concepts
- Gradient Boosting with XGBoost
- One-Hot Encoding for categorical variables
- Feature importance for model interpretability
- Understanding why synthetic data can yield near-perfect scores

## 🔮 Future Improvements
- Hyperparameter tuning (n_estimators, learning_rate, max_depth)
- Comparing against Linear Regression and Random Forest
- Cross-validation for a more robust estimate
