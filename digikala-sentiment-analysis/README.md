# 🛒 Digikala Sentiment Analysis (Persian NLP)

## 📌 Project Overview
A Natural Language Processing (NLP) project that classifies real Persian customer reviews from Digikala (Iran's largest e-commerce platform) to predict whether a reviewer recommends the product.

## 📊 Dataset
- Source: Digikala customer reviews (digikala_text.csv)
- 3,261 records with 3 columns: Text (Persian review), Score (user rating), Suggestion (recommendation label).
- No missing values.

## 🛠 Tech Stack
- Language: Python
- Libraries: Pandas, Scikit-Learn
- Text Vectorization: TfidfVectorizer
- Model: SVC(kernel="linear")

## 🔄 Workflow
1. Data Exploration: Loading 3,261 Persian reviews and inspecting samples with df.info() and head().
2. Feature Extraction: Converting Persian text into numerical vectors using TF-IDF — fitted only on training data to prevent data leakage.
3. Model Training: Training a linear Support Vector Machine (SVM) classifier.
4. Evaluation: Measuring accuracy on the test set.
5. Inference: Predicting the recommendation label for a brand-new, unseen Persian review.

## 📈 Results
- Test Accuracy: 77.18%

## 💡 Key Insights
- A linear kernel works well with high-dimensional TF-IDF text features.
- The model successfully classifies unseen real-world Persian comments.

## 🔮 Future Improvements
- Persian text normalization (handling half-spaces, common spellings).
- Trying MultinomialNB and ensemble models for comparison.
- Using word embeddings (Word2Vec / fastText) for better semantics.
