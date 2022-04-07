import string

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from models.preprocessing import Preprocessor

class NltkTokenStopLemma(Preprocessor):
    _word_net_lemmatizer = WordNetLemmatizer()
    _stopwords = stopwords.words('english')

    def __init__(self) -> None:
        super().__init__()

    def _preprocess(self, raw_text: str) -> str:
        wnl = NltkTokenStopLemma._word_net_lemmatizer
        stopwords = NltkTokenStopLemma._stopwords
        raw_text = raw_text.lower()
        raw_text = raw_text.translate(str.maketrans('', '', string.punctuation))
        raw_tokens = word_tokenize(raw_text)
        tokens = [wnl.lemmatize(raw_token) for raw_token in raw_tokens if raw_token not in stopwords]
        text = ' '.join(tokens)
        return text
