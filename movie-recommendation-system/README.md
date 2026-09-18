# 🎬 Movie Recommendation System (Persian NLP)

## 📌 Project Overview
A Content-Based Movie Recommendation System built with Persian-language data from Filimo (Iranian streaming platform). The user types a movie description or topic, and the system returns the top 5 most similar movies using Cosine Similarity.

## 📊 Dataset
- Source: Filimo movies (movie.csv)
- Columns: Name, Actors, Score, About (Persian synopsis), Genre, Crew, Voted, Type
- Missing values in About handled with fillna("").

## 🛠 Tech Stack
- Language: Python
- Libraries: Pandas, Scikit-Learn
- Text Vectorization: TfidfVectorizer
- Similarity Metric: cosine_similarity

## 🔄 Workflow
1. Data Exploration: Loading the Filimo movie dataset and checking for missing values.
2. Data Cleaning: Filling 522 missing synopses (About) with empty strings.
3. Feature Extraction: Converting Persian movie synopses into TF-IDF vectors.
4. User Input: Taking a free-text topic from the user and vectorizing it with the same fitted TF-IDF.
5. Similarity Ranking: Computing cosine similarity and picking the top 5 matches with argsort()[-5:][::-1].
6. Output: Displaying each recommended movie with its name, score, synopsis, and genre.

## 💡 Key Concepts
- Content-Based Filtering: Recommending based on similarity between item descriptions, not user history.
- Cosine Similarity: Measuring the angle between text vectors to find the closest matches.
- Fully works with Persian free-text queries.

## 🔮 Future Improvements
- Combining multiple text columns (About + Genre + Actors) into richer feature tags.
- Persian text normalization (half-spaces, common spelling variants).
- Building a hybrid recommender with collaborative filtering.
