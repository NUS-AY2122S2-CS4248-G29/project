# Running the code

The code uses Python 3.8.10 which is the Python 3 version in the NUS SOC Compute Cluster. It would probably be useful to use an environment, such as through `virtualenv`, to run this code.

Install the required libraries using `pip`. You can either simply run

`pip install -r requirements.txt`

or install `tensorflow-gpu`, `pandas`, and `numpy` one by one with `pip`.

To train the model, run

`python main.py train`

To train the model while saving checkpoints, run

`python main.py train --save_ckpt_path=[CHECKPOINT PATH]`

To test the model, run

`python main.py test`

To test the model using saved checkpoints, run

`python main.py test --load_ckpt_path=[CHECKPOINT PATH]`

To run the whole thing, run

`python main.py full`

and to run the whole thing while using checkpoints, run

`python main.py full --save_ckpt_path=[CHECKPOINT PATH] --load_ckpt_path=[CHECKPOINT PATH]`
