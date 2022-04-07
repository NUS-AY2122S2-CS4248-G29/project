from typing import Tuple
from typing import Union

import numpy as np

class Dataset:
    def __init__(self) -> None:
        pass

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_validation_data(self) -> Union[Tuple[np.ndarray, np.ndarray], None]:
        pass

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
