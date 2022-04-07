import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from models.preprocessing import NltkTokenStopLemma

class NltkTokenStopLemmaTfidf(NltkTokenStopLemma):
    def __init__(self) -> None:
        super().__init__()
        self._tfidf_vectorizer = TfidfVectorizer()

    def get_vocabulary_size(self) -> int:
        return len(self._tfidf_vectorizer.vocabulary_)

    def adapt(self, raw_text_arr: np.ndarray) -> None:
        raw_text_arr = np.vectorize(self._preprocess)(raw_text_arr)
        self._tfidf_vectorizer = self._tfidf_vectorizer.fit(raw_text_arr)

    def preprocess(self, raw_text_arr: np.ndarray) -> csr_matrix:
        raw_text_arr = np.vectorize(self._preprocess)(raw_text_arr)
        text_sparse_matrix = self._tfidf_vectorizer.transform(raw_text_arr)
        return text_sparse_matrix
