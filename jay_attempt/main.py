import argparse

import numpy as np
import pandas as pd
import tensorflow as tf

SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 32
NUM_EPOCHS = 4

EMBEDDING_DIM = 16
NUM_LSTM_UNITS = 32
NUM_HIDDEN_NODES = 512
NUM_OUTPUT_NODES = 4

def preprocess_data(text_vectorisation_layer, raw_text, raw_label):
    text = text_vectorisation_layer(raw_text)
    label = tf.one_hot(raw_label - 1, 4)
    return text, label

def get_train_data():
    train_df = pd.read_csv('./raw_data/fulltrain.csv', names=['label', 'text'])
    train_df = train_df.sample(frac=1) # shuffle
    train_label_df = train_df['label']
    train_text_df = train_df['text']
    train_label_arr = np.array(train_label_df)
    train_text_arr = np.array(train_text_df)
    return train_label_arr, train_text_arr

def create_train_dataset(train_label_arr, train_text_arr, preprocess_fn):
    raw_train_ds = tf.data.Dataset.from_tensor_slices((train_text_arr, train_label_arr))
    train_ds = raw_train_ds.map(preprocess_fn)
    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE)
    train_ds = train_ds.padded_batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds

def create_model(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, EMBEDDING_DIM,
            # embeddings_regularizer=tf.keras.regularizers.L2()
        ),
        # tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            NUM_LSTM_UNITS,
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # recurrent_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )),
        # tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_HIDDEN_NODES, activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_HIDDEN_NODES, activation='relu',
            kernel_regularizer=tf.keras.regularizers.L2(),
            bias_regularizer=tf.keras.regularizers.L2()
        ),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(
            NUM_OUTPUT_NODES, activation='softmax',
            # kernel_regularizer=tf.keras.regularizers.L2(),
            # bias_regularizer=tf.keras.regularizers.L2()
        )
    ])
    initial_learning_rate = 1e-2
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1e3,
        decay_rate=0.1
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=['accuracy']
    )
    return model

def train(model, train_ds, save_ckpt_path=None):
    callbacks = []
    if save_ckpt_path:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_ckpt_path,
            save_weights_only=True,
            verbose=1
        )
        callbacks.append(cp_callback)
    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks
    )
    return history

def get_test_data():
    test_df = pd.read_csv('./raw_data/balancedtest.csv', names=['label', 'text'])
    test_label_df = test_df['label']
    test_text_df = test_df['text']
    test_label_arr = np.array(test_label_df)
    test_text_arr = np.array(test_text_df)
    return test_label_arr, test_text_arr

def create_test_dataset(preprocess_fn):
    test_label_arr, test_text_arr = get_test_data()
    raw_test_ds = tf.data.Dataset.from_tensor_slices((test_text_arr, test_label_arr))
    test_ds = raw_test_ds.map(preprocess_fn)
    test_ds = test_ds.padded_batch(BATCH_SIZE)
    return test_ds

def evaluate(model, test_ds, load_ckpt_path=None):
    if load_ckpt_path:
        model.load_weights(load_ckpt_path).expect_partial()
    loss, accuracy = model.evaluate(test_ds)
    return loss, accuracy

def main(args):
    train_label_arr, train_text_arr = get_train_data()
    tvl = tf.keras.layers.TextVectorization()
    tvl.adapt(train_text_arr)
    preprocess_data_using_tvl = lambda raw_text, raw_label: preprocess_data(tvl, raw_text, raw_label)
    model = create_model(tvl.vocabulary_size())
    model.summary()
    if args.task in {'train', 'full'}:
        train_ds = create_train_dataset(train_label_arr, train_text_arr, preprocess_data_using_tvl)
        history = train(model, train_ds, save_ckpt_path=args.save_ckpt_path)
    if args.task in {'test', 'full'}:
        test_ds = create_test_dataset(preprocess_data_using_tvl)
        loss, accuracy = evaluate(model, test_ds, load_ckpt_path=args.load_ckpt_path)
        print('Loss:', loss)
        print('Accuracy:', accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='task')

    train_parser = subparser.add_parser('train')
    train_parser.add_argument('--save_ckpt_path', type=str)

    test_parser = subparser.add_parser('test')
    test_parser.add_argument('--load_ckpt_path', type=str)

    full_parser = subparser.add_parser('full')
    full_parser.add_argument('--save_ckpt_path', type=str)
    full_parser.add_argument('--load_ckpt_path', type=str)

    args = parser.parse_args()
    main(args)
