import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, \
        average_precision_score, ndcg_score, \
        label_ranking_average_precision_score
from sklearn.model_selection import train_test_split, KFold
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, \
        GlobalMaxPooling1D, SpatialDropout1D, LSTM, GRU, Flatten, MaxPooling1D, \
        BatchNormalization, ReLU, Bidirectional
from tensorflow.keras.models import Sequential


def init_cnn_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(300, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size = 2))
        #model.add(ReLU())

        model.add(Dropout(.25))
        model.add(Conv1D(200, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size = 2))
        #model.add(ReLU())

        model.add(Dropout(.25))
        model.add(Conv1D(300, 2, activation="relu"))
        model.add(MaxPooling1D(pool_size = 2))
        model.add(BatchNormalization())

        model.add(Dropout(.25))
        model.add(Conv1D(400, 2, activation="relu"))
        model.add(MaxPooling1D(pool_size = 2))
        model.add(BatchNormalization())

        model.add(Dropout(.25))
        model.add(Conv1D(450, 2, activation="relu"))
        model.add(MaxPooling1D(pool_size = 2))
        model.add(BatchNormalization())

        model.add(ReLU())

        model.add(Flatten())
        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


def init_bi_lstm_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(128, dropout=.25)))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(.25))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


def init_bi_lstm_2(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(256, dropout=.25)))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(.25))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


def init_bi_lstm_3(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(.5))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


def init_bi_lstm_cnn_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(128, dropout=.25, return_sequences=True)))
        model.add(Conv1D(384, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


def init_bi_lstm_cnn_2(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(128, dropout=.25, return_sequences=True)))
        model.add(Conv1D(384, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Conv1D(128, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Conv1D(256, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model


def init_bi_lstm_cnn_3(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(128, dropout=.25, return_sequences=True)))
        model.add(Conv1D(512, 5, activation="relu"))
#         model.add(BatchNormalization())
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(.25))
        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model

