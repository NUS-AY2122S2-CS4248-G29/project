#!/usr/bin/env python.

"""
FROM

CS4248 ASSIGNMENT 2 Template

TODO: Modify the variables below.  Add sufficient documentation to cross
reference your code with your writeup.

"""

# Import libraries.  Add any additional ones here.
# Generally, system libraries precede others.
from collections import defaultdict
from distutils.command.sdist import sdist
import numpy as np
import pandas as pd
import scipy
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from utils.nlputil import preprocess_text
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix, hstack

from sklearn.feature_extraction.text import TfidfVectorizer

# TODO: Replace with your Student Number
_STUDENT_NUM = 'result'

class Word2VecVectorizer:
  def __init__(self, model, tfIDF):
    print("Loading in word vectors...")
    self.tfIDF = tfIDF
    self.word_vectors = model
    print("Finished loading in word vectors")

  def fit(self, data):
    if (self.tfIDF):
        tfidf_vectorizer = TfidfVectorizer(norm=None)
        tfidf_vectorizer.fit(data)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf_vectorizer.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf_vectorizer.idf_[i]) for w, i in tfidf_vectorizer.vocabulary_.items()])


  def transform(self, documents):
    sample_vector = self.word_vectors['sample']
    self.dimensions = sample_vector.shape[0]

    all_docs_vector = np.zeros((len(documents), self.dimensions))
    n = 0
    for sentence in documents:
        tokens = sentence.split()
        sentence_words_vector = []
        for word in tokens:
            try:
                word_vector = self.word_vectors[word]
                if (self.tfIDF):
                    word_vector = word_vector * self.word2weight[word]
                sentence_words_vector.append(word_vector)
            except KeyError:
                pass
        if len(sentence_words_vector) > 0:
            sentence_words_vector = np.array(sentence_words_vector)
            all_docs_vector[n] = sentence_words_vector.mean(axis=0)
        n += 1
    return all_docs_vector


  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)

def train_model(model, X_train, y_train, vectorizer):
    ''' TODO: train your model based on the training data '''
    isTraining = 1
    facts_train_tfidf= calculate_feature_set(X_train, isTraining, vectorizer)

    polarities_train_list = y_train.tolist()

    clf = model.fit(facts_train_tfidf, polarities_train_list)

    return clf

def calculate_feature_set(X_train, isTraining, vectorizer):
    facts_train_list = X_train.tolist()

    processed_facts_train = [''] * len(facts_train_list)

    wordnet_lemmatizer = WordNetLemmatizer()

    for idx, doc in enumerate(facts_train_list):
        processed_facts_train[idx] = preprocess_text(doc, lemmatizer=wordnet_lemmatizer, remove_stopwords=False)

    print(processed_facts_train[2])
        
    if (isTraining):
        facts_train_vectors = vectorizer.fit_transform(processed_facts_train)
    else:
        facts_train_vectors = vectorizer.transform(processed_facts_train)

    # facts_train_vectors = csr_matrix(facts_train_vectors)

    # new_feature_col_train = csr_matrix((facts_train_vectors.shape[0], 2), dtype=float)

    # for idx, sentence in enumerate(processed_facts_train):
    #     if 'I' or 'i' in sentence:
    #         new_feature_col_train[idx,0] = sentence.count('I') + sentence.count('i')
    #     if 'think' in sentence:
    #         new_feature_col_train[idx,1] = 1

    # X_train_tfidf = hstack((facts_train_vectors, new_feature_col_train))
            
    return facts_train_vectors

def predict(model, X_test, vectorizer):
    ''' TODO: make your prediction here '''
    isTraining = 0
    X_new_tfidf = calculate_feature_set(X_test, isTraining, vectorizer)
    y_pred = model.predict(X_new_tfidf)

    return y_pred

def generate_result(test, y_pred, filename):
    ''' generate csv file base on the y_pred '''
    test['Verdict'] = pd.Series(y_pred)
    test.drop(columns=['Text'], inplace=True)
    test.to_csv(filename, index=False)

def main():
    ''' load train, val, and test data '''
    train = pd.read_csv('raw_data/balancedtest.csv')
    X_train = train['Text']
    y_train = train['Verdict']
    print("retrieved_csv")
    #model = MultinomialNB()
    model = LogisticRegression(random_state=0, max_iter = 1000)

    isWord2Vec = 1

    if (isWord2Vec):
        word_embeddings = {}
        with open("glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                word_embeddings[word] = vector

        useTfIDF = 1
        vectorizer = Word2VecVectorizer(word_embeddings, useTfIDF)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 1))

    clf = train_model(model, X_train, y_train, vectorizer)

    print("training done")
    # test your model
    y_pred = predict(clf, X_train, vectorizer)

    # Use f1-macro as the metric
    score = f1_score(y_train, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate prediction on test data
    test = pd.read_csv('raw_data/balancedtest.csv')
    X_test = test['Text']
    y_pred = predict(model, X_test, vectorizer)
    generate_result(test, y_pred, _STUDENT_NUM + ".csv")

# Allow the main class to be invoked if run as a file.
if __name__ == "__main__":
    main()