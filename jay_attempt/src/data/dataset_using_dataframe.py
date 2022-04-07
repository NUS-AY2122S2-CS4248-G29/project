import os
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd

from data import Dataset

DIR = os.path.dirname(__file__)

class DatasetUsingDataframe(Dataset):
    def __init__(self) -> None:
        super().__init__()
        pass

    def _split_stratified(self,
        dataframe: pd.DataFrame,
        fraction: float,
        label_column_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        labels = dataframe[label_column_name].unique()
        first_dfs = []
        second_dfs = []
        for label in labels:
            with_label_df = dataframe.loc[dataframe[label_column_name] == label]
            split_index = int(with_label_df.shape[0] * fraction)
            with_label_first_df = with_label_df.iloc[:split_index]
            first_dfs.append(with_label_first_df)
            with_label_second_df = with_label_df.iloc[split_index:]
            second_dfs.append(with_label_second_df)
        first_df = pd.concat(first_dfs)
        second_df = pd.concat(second_dfs)
        return first_df, second_df

    def _get_data_arrays(self, dataframe: pd.DataFrame, column_names: List[str]) -> List[np.ndarray]:
        data_arrs = []
        for column_name in column_names:
            s = dataframe[column_name]
            data_arr = np.array(s)
            data_arrs.append(data_arr)
        return data_arrs

    def _set_training_dataframe(self, training_dataframe: pd.DataFrame) -> None:
        self._training_dataframe = training_dataframe

    def _get_training_dataframe(self) -> pd.DataFrame:
        return self._training_dataframe

    def _set_validation_dataframe(self, validation_dataframe: pd.DataFrame) -> None:
        self._validation_dataframe = validation_dataframe

    def _get_validation_dataframe(self) -> pd.DataFrame:
        return self._validation_dataframe

    def _set_test_dataframe(self, test_dataframe: pd.DataFrame) -> None:
        self._test_dataframe = test_dataframe

    def _get_test_dataframe(self) -> pd.DataFrame:
        return self._test_dataframe
