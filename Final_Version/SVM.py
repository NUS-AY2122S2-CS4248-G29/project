import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import f1_score, accuracy_score

## load data
def load_data(path):
    with open(path+'train_process.data', 'rb') as f:
        X_train = pickle.load(f)
    with open(path+'test_process.data', 'rb') as f:
        X_test = pickle.load(f)

    df_train = pd.read_csv(path+'fulltrain.csv',header=None)
    df_test = pd.read_csv(path+'balancedtest.csv',header=None)
    df_train = df_train.rename(columns={0:"label",1:"text"})
    df_test = df_test.rename(columns={0:"label",1:"text"})
    Y_train = df_train["label"].tolist()
    Y_test = df_test["label"].tolist()
    return X_train, Y_train, X_test, Y_test

# tf-idf vectorization
def vectorize(X_train, X_test):
    tfidf_vec = TfidfVectorizer()
    train = tfidf_vec.fit_transform(X_train).toarray()
    test = tfidf_vec.transform(X_test)
    return train, test

# PCA
def callpca(n_components,X_train,X_test):
    pca = PCA(n_components=n_components)
    pca_result_train = pca.fit_transform(X_train)
    X_test = X_test.toarray()
    pca_result_test = pca.transform(X_test)
    return pca_result_train,pca_result_test


def fit_SVC(X_train,Y_train):
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(X_train, Y_train)
    return clf

def callpredict(clf,X_test):
    Y_pred = clf.predict(X_test)
    return Y_pred

def evaluation(Y_pred,Y_test):
    score1 = f1_score(Y_test, Y_pred, average='macro')
    score2 = accuracy_score(Y_test, Y_pred)
    return score1,score2

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_data("")
    X_train, X_test = vectorize(X_train, X_test)
    n_components = 200
    X_train, X_test = callpca(n_components,X_train,X_test)

    SVC = fit_SVC(X_train,Y_train)
    Y_pred = callpredict(SVC, X_test)
    f1_score, acc_score = evaluation(Y_pred,Y_test)
    print("F1-score:", f1_score)
    print("Accuracy: ", acc_score)
