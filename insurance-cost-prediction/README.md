# 🏥 Insurance Cost Prediction

## 📌 Project Overview
Predicting medical insurance charges based on personal attributes (age, BMI, smoking status, region, etc.) by comparing multiple regression models.

## 📊 Dataset
- 1,338 records with 7 features: age, sex, bmi, children, smoker, region, charges.

## 🛠 Tech Stack
- Language: Python
- Libraries: Pandas, Matplotlib, Scikit-Learn
- Preprocessing: LabelEncoder for categorical columns
- Models Compared: Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbors

## 🔄 Workflow
1. Data Cleaning: Checking for missing values (isna) and removing duplicates (drop_duplicates).
2. EDA: Visual analysis showing that smokers pay, on average, ~4x more in insurance charges than non-smokers.
3. Encoding: Converting categorical features using LabelEncoder.
4. Model Comparison: Training 4 regression models and evaluating them with the R² Score.

## 📈 Model Results
| Model | R² Score |
|---|---|
| Linear Regression | 80.68% |
| Decision Tree | 77.83% |
| Random Forest 🏆 | 88.34% |
| K-Neighbors Regressor | 4.97% |

## 💡 Key Insights
- Random Forest performed best, capturing non-linear relationships in the data.
- Smoking status is the strongest driver of insurance cost.

## 🔮 Future Improvements
- Apply StandardScaler (would drastically improve KNN performance, as it is sensitive to feature scales).
- Hyperparameter tuning with GridSearchCV.
