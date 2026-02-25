from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(text_train)

X_train_tfidf = tfidf_vectorizer.transform(text_train)
X_test_tfidf = tfidf_vectorizer.transform(text_test)

clf_tfidf = LogisticRegression(max_iter=1000, random_state=42)
clf_tfidf.fit(X_train_tfidf, y_train)

clf_tfidf.score(X_test_tfidf, y_test)