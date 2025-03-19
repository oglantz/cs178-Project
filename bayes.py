import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 1234
DATASET_PATH = r'C:\Users\aweso\cs178-files\IMDB Dataset.csv'

data = pd.read_csv(DATASET_PATH)

# Convert sentiment labels to numeric (0 for negative, 1 for positive)
data['sentiment'] = data['sentiment'].map({'negative': 0, 'positive': 1})

X = data['review']
y = data['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.985, random_state=SEED)

tfidf_vectorizer = TfidfVectorizer(
    stop_words='english', 
    max_features=5000, 
    ngram_range=(1, 2))


X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

nb_model = MultinomialNB(alpha=0.5)
nb_model.fit(X_train_tfidf, y_train)

y_pred = nb_model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")

print(f"Classification Report:\n{class_report}")

print(f"Confusion Matrix:\n{conf_matrix}")