import numpy as np
import matplotlib.pyplot as plt
import csv
import re
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, zero_one_loss, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

allLabels = []
allVocab = set()
allDatapoints = []

# Loading the dataset
file_path = "IMDB Dataset.csv"
df = pd.read_csv(file_path)
df=df.sample(20000)
df['sentiment'].replace({'positive':1,'negative':0},inplace=True)

# Apply regex directly to remove special characters and punctuation
df['review'] = df['review'].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)

# Reading all of the vocabulary words
with open(r"aclImdb_v1\aclImdb\imdb.vocab", mode='r', encoding='utf-8') as file:
    allWords = file.readlines()

    for i, vocabWord in enumerate(allWords):
        allVocab.add(vocabWord.strip())

# def removes_stopwords(givenText):
#     stemmedWords = []
#     for word in givenText.split():
#         if word not in stop_words:
#             stemmedWords.append(word)

#     return stemmedWords

# def join_words_back(text_input):
#     return " ".join(text_input)

# df['review']=df['review'].apply(removes_stopwords)
# df['review']=df['review'].apply(join_words_back)

trainingData = df.iloc[:,0:1].values
cv=CountVectorizer(max_features=2500)
trainingData=cv.fit_transform(df['review']).toarray()
labels=df.iloc[:,-1].values
X_train, X_test,y_train,y_test=train_test_split(trainingData,labels,test_size=0.2)



# with open('IMDB Dataset.csv', mode='r', encoding='utf-8') as file:
#     csvFile = csv.reader(file)
    
#     for lines in csvFile:
#         allWords = set(re.findall(r'\b\w+\b', lines[0]))
#         binary_features = [1 if word in allWords else 0 for word in allVocab]
#         allDatapoints.append(binary_features)

#         allLabels.append(lines[1])

# allDatapoints = np.array(allDatapoints)
# X=df.iloc[:,0:1].values

class GaussianBayesClassifier:
    def __init__(self):
        """Initialize the Gaussian Bayes Classifier"""
        self.pY   = []         # class prior probabilities, p(Y=c)
        self.pXgY = []         # class-conditional probabilities, p(X|Y=c)
        self.classes_ = []     # list of possible class values

    def fit(self, X, y):
        """ Fits a Gaussian Bayes classifier with training features X and training labels y.
            X, y : (m,n) and (m,) arrays of training features and target class values
        """
        from sklearn.mixture import GaussianMixture
        self.classes_ = np.unique(y)         # Identify the class labels; then
        for c in self.classes_:              # for each class:
            self.pY.append(np.mean(y==c))    #   estimate p(Y=c) (a float)
            model_c = GaussianMixture(1)     #
            model_c.fit(X[y==c,:])           #   and a Gaussian for p(X|Y=c)
            self.pXgY.append(model_c)        #

    def predict(self, X):
        """ Makes predictions with the nearest centroid classifier on the features in X.
            X : (m,n) array of features for prediction
            Returns: y : (m,) numpy array of predicted labels
        """
        pXY = np.stack(tuple(np.exp(p.score_samples(X)) for p in self.pXgY)).T
        pXY *= np.array(self.pY).reshape(1,-1)         # evaluate p(X=x|Y=c) * p(Y=c)
        pYgX = pXY/pXY.sum(1,keepdims=True)            # normalize to p(Y=c|X=x) (not required)
        return self.classes_[np.argmax(pYgX, axis=1)]  # find the max index & return its class ID

# Plot the decision boundary for your classifier

# Some keyword arguments for making nice looking plots.
plot_kwargs = {'cmap': 'jet',     # another option: viridis
               'response_method': 'predict',
               'plot_method': 'pcolormesh',
               'shading': 'auto',
               'alpha': 0.5,
               'grid_resolution': 100}

figure, axes = plt.subplots(1, 1, figsize=(4,4))

learner = GaussianBayesClassifier()

### YOUR CODE STARTS HERE ###

learner.fit(X_train, y_train)   # Fit "learner" to nych 2-feature data

gbc_y_pred = learner.predict(X_test) # Use "learner" to predict on same data used in training

###  YOUR CODE ENDS HERE  ###

err = zero_one_loss(y_test, gbc_y_pred)
print(f'Gaussian Bayes Error Rate (0/1): {err}')

DecisionBoundaryDisplay.from_estimator(learner, allDatapoints, ax=axes, **plot_kwargs)
axes.scatter(allDatapoints[:, 0], allDatapoints[:, 1], c=allLabels, edgecolor=None, s=12)
axes.set_title(f'Gaussian Bayes Classifier');