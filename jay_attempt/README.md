# Running the code

The code uses Python 3.8.10 which is the Python 3 version in the NUS SOC Compute Cluster. It would probably be useful to use an environment, such as through `virtualenv`, to run this code. The code assumes that the LUN dataset's csv files are in a folder called `raw_data` at the same level as the `src` folder.

Install the required libraries using `pip`. You can either simply run

`pip install -r requirements.txt`

or install `tensorflow-gpu`, `pandas`, `numpy`, `sklearn`, `scipy`, and `nltk`, one by one with `pip`.

To train and evaluate the model, run

`python -m src.main --model=[MODEL_ID] --dataset=[DATASET_ID]`

Model and dataset IDs can be found in `src/models/__init__.py` and `src/data/__init__.py` respectively.

Some other optional parameters for running the code are listed below:

- `--mode=[train | test | full]` is used to choose whether to only train, only test, or do both.
- `--save_filepath=[SAVE_FILEPATH]` is used to specify where to save the model to.
- `--load_filepath=[LOAD_FILEPATH]` is used to specify where to load the model from.
