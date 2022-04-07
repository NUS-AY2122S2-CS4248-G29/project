import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization

EPOCHS = 5
INIT_LR = 3e-5

def read_data(path):
    df = pd.read_csv(path, names=["label", "text"])
    X = df["text"]
    y = df["label"]
    tmp = tf.one_hot(y - 1, depth=4)
    y = np.array(tmp)
    return X, y

def build_model(preprocessor, encoder):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(preprocessor, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoding_layer = hub.KerasLayer(encoder, trainable=True, name='BERT_encoder')
    outputs = encoding_layer(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(4, activation="softmax", name='classifier')(net)
    return tf.keras.Model(text_input, net)

def model_compiler(model, n):
    steps_per_epoch = n
    num_train_steps = steps_per_epoch * EPOCHS
    num_warmup_steps = int(0.1 * num_train_steps)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    metrics = tf.keras.metrics.CategoricalAccuracy()
    optimizer = optimization.create_optimizer(init_lr=INIT_LR,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def model_fitting(model, X_train, y_train, X_val, y_val):
    stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                            min_delta=1e-3,
                                            patience=0)
    model.fit(x=X_train,
              y=y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              callbacks=stop)

    return model

if __name__ == "__main__":
    # read training set
    X_train, y_train = read_data("./raw_data/fulltrain.csv")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

    # build model
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
    clf = build_model(tfhub_handle_preprocess, tfhub_handle_encoder)

    # train model
    clf = model_compiler(clf, X_train.shape[0])
    clf = model_fitting(clf, X_train, y_train, X_val, y_val)

    # read test set
    X_test, y_test = read_data("./raw_data/balancedtest.csv")

    # evaluation
    loss, accuracy = clf.evaluate(X_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')
