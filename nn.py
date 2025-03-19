import pandas as pd
import numpy as np
import re
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


DATASET_PATH = r'C:\Users\aweso\cs178-files\IMDB Dataset.csv'
PROCESSED_DATASET_PATH = r'C:\Users\aweso\cs178-files\processed_dataset.csv'
SEED = 1234

# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')





def preprocess(text):
    """
    Preprocess the input text by removing HTML tags, punctuation, stop words,
    and applying lemmatization.
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text().lower()
    
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters

    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Get English stop words
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words

    lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize the words

    return ' '.join(tokens)  # Join the tokens back into a single string


def startPreprocess():
    dFrame = pd.read_csv(DATASET_PATH)
    dFrame.columns = ['review', 'sentiment']

    dFrame['cleaned_review'] = dFrame['review'].apply(preprocess)  # Apply preprocessing to the reviews
    dFrame['sentiment'] = dFrame['sentiment'].map({'positive': 1, 'negative': 0})  # Map sentiments to binary labels; postive = 1, negative = 0

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Initialize TF-IDF vectorizer with a maximum of 5000 features
    X_tfidf = tfidf_vectorizer.fit_transform(dFrame['cleaned_review'])
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())  # Convert the TF-IDF matrix to a DataFrame

    dFrame.drop(columns=['review'], inplace=True)  # Drop the original review column
    dFrame = pd.concat([dFrame, X_tfidf_df], axis=1)  # Concatenate the TF-IDF features with the DataFrame

    dFrame.to_csv(PROCESSED_DATASET_PATH, index=False)  # Save the processed DataFrame to a CSV file
    print(dFrame.head())  # Print the first few rows of the processed DataFrame


# if __name__ == "__main__":
#     # startPreprocess() # ONLY NEED TO RUN ONCE

#     dFrame = pd.read_csv(PROCESSED_DATASET_PATH)  # Read the processed dataset


#     # Drop missing values, if exist
#     dFrame.dropna(subset=['review'], inplace=True)

#     dFrame['review'] = dFrame['review'].astype(str)  # Ensure the review column is of type string


#     X_train, X_test, y_train, y_test = train_test_split(dFrame["review"], dFrame["sentiment"], test_size=0.2, random_state=42)

#     tfidf_vectorizer = TfidfVectorizer(max_features=10000,
#                                        ngram_range=(1, 2),
#                                        sublinear_tf=True)  # Initialize TF-IDF vectorizer
#     X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Fit and transform the training data
#     X_test_tfidf = tfidf_vectorizer.transform(X_test)  # Transform the test data


#     mlp_model = MLPClassifier(
#         hidden_layer_sizes=(512, 256),  # 3 hidden layers
#         activation='relu',  # Activation function
#         solver='adam',  # Optimization algorithm
#         alpha=0.0005,  # Regularization parameter
#         learning_rate_init=0.001,  # Initial learning rate
#         batch_size=64,
#         max_iter=15,
#         random_state=SEED,
#         verbose=True
#     )

#     mlp_model.fit(X_train_tfidf, y_train)  # Train the model on the training data
#     y_pred = mlp_model.predict(X_test_tfidf)  

#     accuracy = accuracy_score(y_test, y_pred)  # accuracy

#     print(f"Test Accuracy: {accuracy:.4f}")  # Print the test accuracy
#     print("Classification Report:\n", classification_report(y_test, y_pred))  # Print classification report

#     print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


glove_path = r'C:\Users\aweso\cs178-files\glove.6B.50d.txt'  # Path to GloVe embeddings file
word_embeddings = {}

def load_embeddings():
    """
    Load the GloVe embeddings and return a dictionary mapping words to their embeddings.
    """
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # First value is the word
            vector = np.array(values[1:], dtype=np.float32)  # Rest are the vector values
            word_embeddings[word] = vector




if __name__ == "__main__":
    load_embeddings()  # Load the GloVe embeddings
    dFrame = pd.read_csv(DATASET_PATH)  # Read the dataset

    dFrame.dropna(subset=['review'], inplace=True)
    dFrame['review'] = dFrame['review'].astype(str)

    dFrame['sentiment'] = dFrame['sentiment'].map({'positive': 1, 'negative': 0})


    # Load Pretrained GloVe Embeddings (Word2Vec Format)

    def get_embedding(text, embedding_dim=50):
        words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
        embeddings = [word_embeddings[word] for word in words if word in word_embeddings]  # Use only known words
        if len(embeddings) == 0:
            return np.zeros(embedding_dim)  # Return zero vector if no valid words
        return np.mean(embeddings, axis=0)  # Average word embeddings

    X_emb = np.array([get_embedding(review) for review in dFrame['review']])  # Get embeddings for all reviews
    y = dFrame['sentiment'].values  # Get the sentiment labels

    

    # print("First 5 reviews converted to embeddings:")
    # print(X_emb[:5])
    # print("Are all embeddings zero?", np.all(X_emb == 0))


    X_train, X_test, y_train, y_test = train_test_split(X_emb, y, test_size=0.6, random_state=SEED)  # Split the data into training and test sets
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(1024, 512, 256, 128),  # 4 hidden layers
        activation='relu',  # Activation function
        solver='adam',  # Optimization algorithm
        alpha=0.00005,  # Regularization parameter
        learning_rate_init=0.001,  # Initial learning rate
        batch_size=32,
        max_iter=100,
        random_state=SEED,
        verbose=True
    )

    mlp_model.fit(X_train, y_train)

    # Make predictions
    y_pred = mlp_model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Show classification report
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Show confusion matrix
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))