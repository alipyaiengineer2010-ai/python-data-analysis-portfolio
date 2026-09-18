# 📧 Spam / Ham Classifier (NLP + SVM)

## 📌 Project Overview
A binary text classification system that detects spam messages using the classic SMS Spam Collection dataset. Text is vectorized with TF-IDF and classified with a Linear SVC, achieving 99% accuracy with perfect precision on the spam class.

## 📊 Dataset
- Source: SMS Spam Collection (email.csv)
- ~5,573 messages labeled ham (4,825) or spam (747)
- Data Quality Fix: Detected and removed one corrupted row ({"mode":"full"}) discovered during EDA with value_counts().

## 🛠 Tech Stack
- Language: Python
- Libraries: Pandas, Scikit-Learn
- Model: SVC(kernel="linear")
- Text Vectorization: TfidfVectorizer
- Encoding: LabelEncoder (+ inverse_transform for human-readable predictions)

## 🔄 Workflow
1. EDA: Inspecting class distribution and spotting a corrupted row.
2. Cleaning: Filtering out the invalid category row.
3. Encoding: ham/spam → 0/1 with LabelEncoder.
4. Split: 80/20 train-test split with random_state=42.
5. Vectorization: TF-IDF fitted only on training data (preventing data leakage), then applied to the test set.
6. Training & Evaluation: Linear SVC + classification_report.
7. Inference: Predicting on custom messages and decoding labels back with inverse_transform.

## 📈 Results
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| ham (0)  | 0.99 | 1.00 | 1.00 |
| spam (1) | 1.00 | 0.95 | 0.98 |
| Accuracy | | | 0.99 |

## 💡 Key Concepts
- TF-IDF feature extraction for text
- Avoiding data leakage (fit_transform on train, transform on test)
- Evaluating imbalanced data with precision/recall, not accuracy alone
- Inverse-transforming predictions back to original labels

## 🔮 Future Improvements
- Stratified train-test split (stratify=y) for better class balance
- Comparing Naive Bayes, Logistic Regression, and SVC
- Building a Pipeline + deploying as a web app (Streamlit)
