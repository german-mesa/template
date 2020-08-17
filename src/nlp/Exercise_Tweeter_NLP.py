# Sentiment analysis with Tweets
# - Tweets have been annotated 0-4 (Negative-Positive) - Field 0
# - Tweet test - Field 5
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.keras as keras
import tensorflow.keras.utils as utils

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

TRAINING_SIZE = 160000
TRAINING_SPLIT = 0.9

MAX_LENGTH = 16
PADDING_TYPE = 'post'
TRUNCATING_TYPE = 'post'
EMBEDDING_DIM = 100

NUM_EPOCHS = 50


def get_tweet_dataset(url):
    corpus = []

    filename = os.path.basename(url)

    utils.get_file(fname=filename,
                   origin=url,
                   cache_dir='.')

    print('Opening CSV file to get Tweet data...')
    filename = os.path.join(os.getcwd(), 'datasets', filename)

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            corpus_item = [row[5]]

            if row[0] == '0':
                corpus_item.append(0)
            else:
                corpus_item.append(1)

            corpus.append(corpus_item)

    print(f'There are {len(corpus)} tweets available')
    return corpus


def get_sample_dataset():
    sentences = []
    labels = []

    # Get data from CSV file
    corpus = get_tweet_dataset("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv")

    # Get a random portion of data from this file
    random.shuffle(corpus)
    for index in range(TRAINING_SIZE):
        sentences.append(corpus[index][0])
        labels.append(corpus[index][1])

    return sentences, labels


def get_embedding_weights(word_index):
    # Get word embeddings from a previously trained model
    embeddings_index = get_word_embeddings()

    # Create weight matrix
    embeddings_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM));
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word);
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector;

    return embeddings_matrix


def get_word_embeddings():
    embeddings_index = {}

    filename = os.path.join(os.getcwd(), 'datasets', 'glove.6B.100d.txt')

    with open(filename, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings_index[word] = np.asarray(values[1:], dtype='float32')

    return embeddings_index


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def main():
    # Get a portion of data from CSV file
    sentences, labels = get_sample_dataset()

    # Tokenizing items
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=MAX_LENGTH, padding=PADDING_TYPE, truncating=TRUNCATING_TYPE)

    vocabulary_dim = len(tokenizer.word_index)

    # Get embeddings from a previously trained model
    embeddings_matrix = get_embedding_weights(tokenizer.word_index)

    # Split in training and testing data
    split = int(TRAINING_SPLIT * TRAINING_SIZE)

    training_padded = padded[0:split]
    training_labels = labels[0:split]

    testing_padded = padded[split:TRAINING_SIZE]
    testing_labels = labels[split:TRAINING_SIZE]

    # Define model
    model = keras.Sequential(
        [
            keras.layers.Embedding(vocabulary_dim + 1,              # Include Out-of-vocabulary
                                   EMBEDDING_DIM,
                                   input_length=MAX_LENGTH,
                                   weights=[embeddings_matrix],
                                   trainable=False),
            keras.layers.Dropout(0.2),
            keras.layers.Conv1D(64, 5, activation=keras.activations.relu),
            keras.layers.MaxPooling1D(4),
            keras.layers.LSTM(64),
            keras.layers.Dense(1, activation=keras.activations.sigmoid)
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    model.summary()

    # Train model
    history = model.fit(
        np.array(training_padded),
        np.array(training_labels),
        validation_data=(np.array(testing_padded), np.array(testing_labels)),
        epochs=NUM_EPOCHS,
        verbose=2
    )

    # Plot results
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')


if __name__ == '__main__':
    main()
