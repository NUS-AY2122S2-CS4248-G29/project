from models.model import Model
from models.tensorflow_models.base_tensorflow_model import TensorFlowModel
from models.tensorflow_models.numpy_tensorflow_model import NumpyTensorFlowModel
from models.tensorflow_models.csr_matrix_tensorflow_model import CsrMatrixTensorFlowModel
from models.tensorflow_models.sequential_lstm_model import SequentialLstmModel
from models.tensorflow_models.subclassed_lstm_model import SubclassedLstmModel
from models.tensorflow_models.residual_lstm_model import ResidualLstmModel
from models.tensorflow_models.nltk_rnn_model import NltkRnnModel
from models.tensorflow_models.nltk_tfidf_model import NltkTfidfModel
from models.tensorflow_models.transformer_model import TransformerModel

models = {
    'sequential_lstm_model': SequentialLstmModel,
    'subclassed_lstm_model': SubclassedLstmModel,
    'residual_lstm_model': ResidualLstmModel,
    'nltk_tfidf_model': NltkTfidfModel,
    'nltk_rnn_model': NltkRnnModel,
    'transformer_model': TransformerModel
}
