import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# read data
train_df = pd.read_csv("./fulltrain.csv", names = ["label", "text"])
test_df = pd.read_csv("./balancedtest.csv", names = ["label", "text"])

X_train = train_df.loc[:, "text"]
y_train = train_df.loc[:, "label"]
X_test = test_df.loc[:, "text"]
y_test = test_df.loc[:, "label"]

# preprocessing
def preprocessing(X):
    wnl = WordNetLemmatizer()
    X_lower = X.str.lower() # lowercase
    X_processed = []
    for x in X_lower:
        x_token = word_tokenize(x) # tokenization
        x_word = []
        for t in x_token:
            if t in stopwords.words("english"): # stopword removal
                continue
            if re.match(r"[^0-9a-zA-Z]", t): # punctuation removal
                continue
            x_word.append(t)
        x_lemma = [wnl.lemmatize(w) for w in x_word] # lemmatization
        x_clean = " ".join(x_lemma)
        X_processed.append(x_clean)
    return X_processed

X_train = preprocessing(X_train)
X_test = preprocessing(X_test)

# vectorization
# tf-idf
def vectorize(X_train, X_test):
    tfidf_vec = TfidfVectorizer()
    train = tfidf_vec.fit_transform(X_train).toarray()
    test = tfidf_vec.transform(X_test)
    return train, test

train, test = vectorize(X_train, X_test)

# model
def train_pred(train, test, y_train):
    clf = LogisticRegression(random_state=0, max_iter=500)
    clf.fit(train, y_train)
    y_pred = clf.predict(test)
    return y_pred

y_pred = train_pred(train, test, y_train)

score1 = f1_score(y_test, y_pred, average='macro')
score2 = accuracy_score(y_test, y_pred)
print("F1-score:", score1)
print("Accuracy: ", score2)





