import os
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from data import DatasetUsingDataframe

DIR = os.path.dirname(__file__)

VALIDATION_FRACTION = 0.2

class BuptGamma(DatasetUsingDataframe):
    def __init__(self) -> None:
        super().__init__()
        train_df = pd.read_csv(os.path.join(DIR, '..', '..', 'raw_data', 'fulltrain.csv'), names=['label', 'text'])
        train_df = train_df.astype({'label': 'int32'})
        train_df['label'] -= 1
        self._set_training_dataframe(train_df)
        test_df = pd.read_csv(os.path.join(DIR, '..', '..', 'raw_data', 'balancedtest.csv'), names=['label', 'text'])
        test_df['label'] -= 1
        self._set_test_dataframe(test_df)

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        train_df = self._get_training_dataframe()
        text_arr, label_arr = self._get_data_arrays(train_df, ['text', 'label'])
        return text_arr, label_arr

    def get_validation_data(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        return None

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        test_df = self._get_test_dataframe()
        text_arr, label_arr = self._get_data_arrays(test_df, ['text', 'label'])
        return text_arr, label_arr
