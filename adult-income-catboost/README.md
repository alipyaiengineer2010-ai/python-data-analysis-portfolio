# 💰 Adult Income Prediction with CatBoost

## 📌 Project Overview
Predicting whether a person earns more than $50K/year from US Census data using CatBoost — with native categorical feature handling (no One-Hot Encoding needed), class imbalance handling, and automatic overfitting detection.

## 📊 Dataset
- Source: Adult Census Income (32,561 records, 15 features)
- Features: Age, Workclass, Education, Occupation, Relationship, Race, Sex, Capital Gain/Loss, Hours per Week, Native Country, ...
- Target: Income (>50K → 1, <=50K → 0)
- Cleaning: The dataset encodes missing values as "?" — replaced with pd.NA and dropped (replace + dropna).
- Class imbalance: 22,654 low-income vs. 7,508 high-income samples → handled with balanced class weights.

## 🛠 Tech Stack
- Language: Python
- Libraries: Pandas, NumPy, Scikit-Learn, CatBoost, Matplotlib
- Model: CatBoostClassifier (1000 iterations, depth=6, lr=0.1, eval_metric=F1)
- Key features: cat_features for native categorical handling, class_weights, early_stopping_rounds=100

## 🔄 Workflow
1. EDA: head(), info() (15 columns, int64/object dtypes).
2. Cleaning: "?" → NA → dropped; target encoded with a lambda on ">50K".
3. Split: 80/20 train-test (random_state=42).
4. Imbalance: compute_class_weight("balanced") → weights [0.666, 2.008].
5. Training: CatBoost with native categorical features + eval set.
6. Early stopping: Overfitting detector stopped training at iteration 271 (best F1 = 0.845) and shrank the model automatically.
7. Evaluation: Accuracy, classification report, and validation F1 curve via get_evals_result().

## 📈 Results
- Accuracy: 83.2% — competitive for this benchmark dataset.
- High-income class (1): Precision 0.62, Recall 0.86, F1 0.72 — the class weights successfully boosted recall for the minority class.
- Learning curve: validation F1 rose from 0.81 to ~0.845 and plateaued — a healthy, stable curve.

## 💡 Key Concepts
- CatBoost's native categorical feature handling (no manual encoding)
- Handling imbalanced classification with class weights
- Early stopping / overfitting detection
- Choosing Recall vs. Precision trade-offs for the minority class

## 🔮 Future Improvements
- Dropping the fnlwgt column (census sampling weight, not informative for ML)
- Hyperparameter tuning with grid_search
- Comparing CatBoost vs. XGBoost with One-Hot Encoding
- SHAP values for interpretability
