import numpy as np
import tensorflow as tf
from tensorflow.distribute import MirroredStrategy
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Conv1D, \
        GlobalMaxPooling1D, SpatialDropout1D, LSTM, GRU, Flatten, MaxPooling1D, \
        BatchNormalization, ReLU, Bidirectional, Concatenate
from tensorflow.keras import Sequential, Model, Input


def cnn_1(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    embedding_matrix = kwargs["embedding_matrix"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            mask_zero=True,
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(300, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size = 2))

        model.add(Dropout(.25))
        model.add(Conv1D(200, 2, activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size = 2))

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


def cnn_2(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    embedding_matrix = kwargs["embedding_matrix"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            mask_zero=True,
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


def cnn_3(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    embedding_matrix = kwargs["embedding_matrix"]
    filters_1 = kwargs["filters_1"]
    filters_2 = kwargs["filters_2"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            mask_zero=True,
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(filters_1, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(filters_2, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Flatten())

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def bi_lstm_1(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    embedding_matrix = kwargs["embedding_matrix"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            mask_zero=True,
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def cnn_bi_lstm_1(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    filters_1 = kwargs["filters_1"]
    filters_2 = kwargs["filters_2"]
    embedding_matrix = kwargs["embedding_matrix"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            mask_zero=True,
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(filters_1, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(filters_2, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def cnn_bi_lstm_2(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    filters_1 = kwargs["filters_1"]
    filters_2 = kwargs["filters_2"]
    embedding_matrix = kwargs["embedding_matrix"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        model = Sequential()

        model.add(Embedding(
            input_dim=n_vocabulary,
            output_dim=n_embedding,
            embeddings_initializer=Constant(embedding_matrix),
            mask_zero=True,
            input_length=n_sequence,
            trainable=False
        ))

        model.add(Conv1D(filters_1, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Conv1D(filters_2, 2))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(MaxPooling1D(2))
        model.add(Dropout(.2))

        model.add(Bidirectional(LSTM(256, dropout=.2, return_sequences=True)))
        model.add(Bidirectional(LSTM(128, dropout=.2)))

        model.add(Dense(n_labels, activation="sigmoid"))
        model.compile(loss=loss, optimizer="adam")
        return model


def split_cnn_bi_lstm_1(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    filters_1 = kwargs["filters_1"]
    filters_2 = kwargs["filters_2"]
    embedding_matrix = kwargs["embedding_matrix"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        input_layer = Input(n_sequence,)
        embedding_layer = Embedding(n_vocabulary, n_embedding,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    mask_zero=True, trainable=True)(input_layer)
        cnn_list = []
        for kernel_size in [2, 3]:
            x = Conv1D(filters_1, kernel_size, padding="same")(embedding_layer)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(.2)(x)
            x = Conv1D(filters_2, kernel_size, padding="same")(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(.2)(x)
            cnn_list.append(x)

        x = Concatenate()(cnn_list)

        x = Bidirectional(LSTM(256, dropout=.2))(x)

        x = Dense(n_labels, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss=loss, optimizer="adam")
        return model


def split_cnn_bi_lstm_2(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    filters_1 = kwargs["filters_1"]
    filters_2 = kwargs["filters_2"]
    loss = kwargs["loss"]
    embedding_matrix_1 = kwargs["embedding_matrix_1"]
    embedding_matrix_2 = kwargs["embedding_matrix_2"]

    with MirroredStrategy().scope():
        input_layer = Input(n_sequence,)

        cnn_list = []
        for embedding_matrix in [embedding_matrix_1, embedding_matrix_2]:
            x = Embedding(n_vocabulary, n_embedding,
                          embeddings_initializer=Constant(embedding_matrix),
                          mask_zero=True, trainable=False)(input_layer)
            x = Conv1D(filters_1, 2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(.2)(x)
            x = Conv1D(filters_2, 2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(.2)(x)
            cnn_list.append(x)

        x = Concatenate()(cnn_list)

        x = Bidirectional(LSTM(256, dropout=.2))(x)

        x = Dense(n_labels, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss=loss, optimizer="adam")
        return model

def split_cnn_bi_lstm_3(n_vocabulary, n_embedding, n_sequence, n_labels, **kwargs):
    filters_1 = kwargs["filters_1"]
    filters_2 = kwargs["filters_2"]
    embedding_matrix_1 = kwargs["embedding_matrix_1"]
    embedding_matrix_2 = kwargs["embedding_matrix_2"]
    loss = kwargs["loss"]

    with MirroredStrategy().scope():
        input_layer = Input(n_sequence,)

        cnn_list = []
        for embedding_matrix in [embedding_matrix_1, embedding_matrix_2]:
            x = Embedding(n_vocabulary, n_embedding,
                          embeddings_initializer=Constant(embedding_matrix),
                          mask_zero=True, trainable=False)(input_layer)
            x = Conv1D(filters_1, 2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(.2)(x)
            x = Conv1D(filters_2, 2)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = MaxPooling1D(2)(x)
            x = Dropout(.2)(x)
            cnn_list.append(x)

        x = Concatenate()(cnn_list)

        x = Bidirectional(LSTM(256, dropout=.2, return_sequences=True))(x)
        x = Bidirectional(LSTM(128, dropout=.2))(x)

        x = Dense(n_labels, activation="sigmoid")(x)
        model = Model(inputs=input_layer, outputs=x)
        model.compile(loss=loss, optimizer="adam")
        return model
