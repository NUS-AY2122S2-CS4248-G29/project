from keras.preprocessing import sequence
from keras.preprocessing import text
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, LSTM
from keras.layers import Conv1D, Flatten, MaxPooling1D
import tensorflow as tf
from google.colab import drive
import os
import pickle

VOCAB_SIZE = 1000
MAXLEN = 1000
EMBEDDING_DIMS = 50
FILTERS = 32
KERNEL_SIZE = 3
HIDDEN_DIMS = 250
BATCH_SIZE = 10000
EPOCHS = 20
NUM_OUTPUT_NODES = 4

drive.mount('/content/drive')
print(os.listdir('drive/MyDrive/gp29'))
print(os.getcwd())
data_dir = "/content/drive/MyDrive/gp29/"

with open(data_dir +'preprocessed_data/train_process.data', 'rb') as f:
    X_train = pickle.load(f)
with open(data_dir +'preprocessed_data/test_process.data', 'rb') as f:
    X_test = pickle.load(f)

train = pd.read_csv(data_dir+'raw_data/fulltrain.csv', header=None, names=["label","news"])
test = pd.read_csv(data_dir+'raw_data/balancedtest.csv', header=None, names=["label","news"])
X_train = train["news"]
y_train = train["label"]
X_test = test["news"]
y_test = test["label"]

tokenizer = text.Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_matrix(X_train)
X_test = tokenizer.texts_to_matrix(X_test)

X_train = sequence.pad_sequences(X_train,maxlen=MAXLEN)
X_test = sequence.pad_sequences(X_test,maxlen=MAXLEN)

y_train = tf.one_hot(y_train - 1, 4)
y_test = tf.one_hot(y_test - 1, 4)

model = Sequential()
model.add(Embedding(VOCAB_SIZE,
                   KERNEL_SIZE,
                   input_length=MAXLEN))
model.add(Conv1D(FILTERS,
                KERNEL_SIZE,
                padding='valid',
                activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(HIDDEN_DIMS, activation='relu'))
model.add(Dense(NUM_OUTPUT_NODES, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")
])

model.fit(X_train,y_train,
         batch_size=BATCH_SIZE,
         epochs=EPOCHS,
         validation_data=(X_test,y_test),)
