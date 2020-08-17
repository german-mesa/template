import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.utils as utils

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

EPOCHS = 100


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


def test_model(model, tokenizer, seed_text, next_words=100, max_sequence_size=100):
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
    # Get data from file
    corpus = get_data_from_file("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/irish-lyrics-eof.txt")

    # Create word index using tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    # Prepare sequences to guess next words
    input_sequences = []

    sequences = tokenizer.texts_to_sequences(corpus)

    for sequence in sequences:
        for digit in range(1, len(sequence)):
            input_sequences.append(sequence[:digit + 1])

    # Padding sequences
    max_sequence_size = max([len(x) for x in input_sequences])
    padded_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_size))

    # Choose xs and ys
    xs = padded_sequences[:, :-1]
    ys = tf.keras.utils.to_categorical(padded_sequences[:, -1], num_classes=total_words)

    # Define model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_size - 1),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)),
            tf.keras.layers.Dense(total_words, activation=tf.keras.activations.softmax)
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
        xs,
        ys,
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
                   seed_text="I've got a bad feeling about this",
                   max_sequence_size=max_sequence_size)
    )


if __name__ == '__main__':
    main()
