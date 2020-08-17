import os
import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.utils as utils

EPOCHS = 100
EMBEDDINGS_DIM = 100


def get_data_from_file(url):
    filename = os.path.basename(url)

    utils.get_file(fname=filename,
                   origin=url,
                   cache_dir='.')

    filename = os.path.join(os.getcwd(), 'datasets', filename)
    data = open(filename).read()
    corpus = data.lower().split('\n')

    return corpus


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show()


def test_model(model, tokenizer, seed_text, next_words=100, max_sequence_size=10):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_size - 1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text


def main():
    # Get sonnets from file
    corpus = get_data_from_file("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt")

    # Tokenize corpus
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    # Create input sequences
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            input_sequences.append(token_list[:i + 1])

    # Pad input sequences
    max_sequence_size = max([len(x) for x in input_sequences])
    padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_size))

    predictors, labels = padded_sequences[:, :-1], padded_sequences[:, -1]
    labels = utils.to_categorical(labels, num_classes=total_words)

    # Prepare model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(total_words,
                                      EMBEDDINGS_DIM,
                                      input_length=max_sequence_size - 1),
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(150, return_sequences=True)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(total_words / 2,
                                  activation=tf.keras.activations.relu,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(total_words,
                                  activation=tf.keras.activations.softmax)
        ]
    )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        predictors,
        labels,
        epochs=EPOCHS,
        verbose=2
    )

    # Plot accuracy & loss
    plot_graphs(history, 'loss')
    plot_graphs(history, 'accuracy')

    # Test our model
    print(
        test_model(model=model,
                   tokenizer=tokenizer,
                   seed_text="Help me Obi Wan Kenobi, you're my only hope",
                   max_sequence_size=max_sequence_size)
    )


if __name__ == '__main__':
    main()
