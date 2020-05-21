import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, \
        GlobalMaxPooling1D, SpatialDropout1D, LSTM, GRU, Flatten, MaxPooling1D, \
        BatchNormalization, ReLU, Bidirectional
from tensorflow.keras.models import Sequential


def init_cnn_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
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
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_2(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(200, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(300, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(400, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(500, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(600, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Flatten())

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_bi_lstm_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(128, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_bi_lstm_2(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(400, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(500, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_2(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(512, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(512, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_3(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(512, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(512, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(512, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_4(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(512, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_5(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(400, 3))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(500, 3))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_6(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(400, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_7(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(400, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(500, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(512, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_lstm_8(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(400, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(500, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(128, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_lstm_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(400, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(500, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(LSTM(256, dropout=.2))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def init_cnn_bi_gru_1(embedding_matrix, n_vocabulary, n_embedding, n_sequence, n_labels, loss):
    model = Sequential()

    model.add(Embedding(
        input_dim=n_vocabulary,
        output_dim=n_embedding,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=n_sequence,
        trainable=False
    ))

    model.add(Conv1D(400, 2))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(2))
    model.add(Dropout(.2))

    model.add(Conv1D(500, 2))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(MaxPooling1D(2))
    model.add(Dropout(.2))

    model.add(Bidirectional(GRU(256, dropout=.2)))

    model.add(Dense(n_labels, activation="sigmoid"))
    model.compile(loss=loss, optimizer="adam")
    return model
