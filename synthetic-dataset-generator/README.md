# 🧪 Synthetic Dataset Generator & Feature Engineering

## 📌 Project Overview
A complete pipeline that generates a synthetic tabular dataset with NumPy, simulates real-world data quality issues (missing values), and applies professional data cleaning and feature engineering techniques with Pandas — turning raw random data into an ML-ready dataset.

## 🛠 Tech Stack
- Language: Python
- Libraries: NumPy, Pandas, Matplotlib
- Environment: Jupyter Notebook (VS Code)

## 🔄 Workflow
1. Synthetic Data Generation: Creating 50 samples (age, children, marital status, salary) with np.random and a fixed seed(42) for reproducibility.
2. Simulating Missing Data: Injecting NaN values manually to replicate real-world data quality problems.
3. Custom Cleaning Functions: fill_median() (median imputation) and encode() (binary encoding of yes/no via map + lambda).
4. Feature Engineering:
   - Age Binning: Grouping ages into young / adult / middle / senior with pd.cut.
   - Target Creation: Building a binary is_high_income label based on the median salary — converting a regression problem into classification.
5. Visualization: Age distribution histogram with Matplotlib.

## 💡 Key Concepts
- Reproducibility with np.random.seed
- Median imputation vs. dropping rows
- Categorical encoding for ML models
- Binning continuous features (pd.cut)
- Derived target variables (median-based income labeling)

## 🔮 Future Improvements
- Vectorized target creation with .astype(int) instead of a loop
- Adding correlations between columns for more realistic synthetic data
- Training a classifier on the generated dataset
