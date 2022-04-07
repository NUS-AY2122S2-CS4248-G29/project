import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

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

# tf-idf
def vectorize(X_train, X_test):
    tfidf_vec = TfidfVectorizer()
    train = tfidf_vec.fit_transform(X_train).toarray()
    test = tfidf_vec.transform(X_test).toarray()
    return train, test

# model
def train_pred(clf, train, test, y_train):
    clf.fit(train, y_train)
    y_pred = clf.predict(test)
    return y_pred

if __name__ == "__main__":
    # load data
    train_df = pd.read_csv("./fulltrain.csv", names=["label", "text"])
    test_df = pd.read_csv("./balancedtest.csv", names=["label", "text"])

    X_train = train_df.loc[:, "text"]
    y_train = train_df.loc[:, "label"]
    X_test = test_df.loc[:, "text"]
    y_test = test_df.loc[:, "label"]

    # preprocessing
    X_train = preprocessing(X_train)
    X_test = preprocessing(X_test)

    # vectorization
    train, test = vectorize(X_train, X_test)

    # logistic regression
    lr = LogisticRegression(random_state=0, max_iter=500)
    y_pred = train_pred(lr, train, test, y_train)
    print("-- Logistic --")
    confusion = confusion_matrix(y_pred, y_test)
    print(confusion)

    # feature engineering
    pca = PCA(n_components=200)
    pca_train = pca.fit_transform(train)
    pca_test = pca.transform(test)

    # svm
    svc = SVC(decision_function_shape='ovo')
    y_pred = train_pred(svc, pca_train, pca_test, y_train)
    print("-- SVM --")
    confusion = confusion_matrix(y_pred, y_test)
    print(confusion)





