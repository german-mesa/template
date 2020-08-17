import os
import io
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

TRAINING_SPLIT = 0.8
OOV_TOKEN = '<OOV>'
VOCABULARY_SIZE = 1000
EMBEDDING_SIZE = 16
MAX_PADDING_LENGTH = 120

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]


# Plot results
def plot_visualize_training(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, len(history.history['loss']), 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, len(history.history['loss']), 1))

    plt.legend(loc='best', shadow=True)
    plt.tight_layout()
    plt.show()


def save_vocabulary(filename, vocabulary):
    reverse_word_index = dict([(value, key) for (key, value) in vocabulary.items()])

    with open(filename, 'w', encoding='utf-8') as f:
        for index in range(1, VOCABULARY_SIZE):
            f.write(reverse_word_index[index] + "\n")


def save_embeddings(filename, embeddings):
    with open(filename, 'w', encoding='utf-8') as f:
        weights = embeddings.get_weights()[0]

        for index in range(1, VOCABULARY_SIZE):
            f.write('\t'.join([str(x) for x in weights[index]]) + "\n")


def get_data_from_json():               # Not used
    # Open article data store
    with open(os.path.join(os.getcwd(), 'datasets', 'sarcasm.json')) as f:
        data_store = json.load(f)

    # Prepare material for analysis
    headlines = []
    labels = []
    links = []

    for item in data_store:
        headlines.append(item['headline'])
        labels.append(item['is_sarcastic'])
        links.append(item['article_link'])


def get_text_data_from_csv(filename):
    sentences = []
    labels = []

    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)

        for row in reader:
            labels.append(row[0])
            sentence = row[1]
            for word in stopwords:
                sentence = sentence.replace(' ' + word + ' ', ' ')

            sentences.append(sentence)

    return sentences, labels


def main():
    # Get data from CSV file
    sentences, labels = get_text_data_from_csv(os.path.join(os.getcwd(), 'datasets', 'bbc-text.csv'))
    print(f'Number of sentences in file {len(labels)}')

    train_size = int(len(sentences) * TRAINING_SPLIT)

    train_sentences = sentences[:train_size]
    train_labels = labels[:train_size]
    print(f'Number of sentences for training {len(train_labels)}')

    validation_sentences = sentences[train_size:]
    validation_labels = labels[train_size:]
    print(f'Number of sentences for validation {len(validation_labels)}')

    # Tokenize and prepare sequences - For sentences
    sentence_tokenizer = Tokenizer(num_words=VOCABULARY_SIZE, oov_token=OOV_TOKEN)
    sentence_tokenizer.fit_on_texts(train_sentences)
    print(f'Number of tokens in sentence tokenizer {len(sentence_tokenizer.word_index)}')

    train_sequences = sentence_tokenizer.texts_to_sequences(train_sentences)
    train_padded_sequences = pad_sequences(train_sequences, maxlen=MAX_PADDING_LENGTH, padding='post')

    validation_sequences = sentence_tokenizer.texts_to_sequences(validation_sentences)
    validation_padded_sequences = pad_sequences(validation_sequences, maxlen=MAX_PADDING_LENGTH, padding='post')

    # Tokenize and prepare sequences - For labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    print(f'Number of tokens in label tokenizer {len(label_tokenizer.word_index)}')

    train_labels_sequences = np.array(label_tokenizer.texts_to_sequences(train_labels))
    validation_labels_sequences = np.array(label_tokenizer.texts_to_sequences(validation_labels))

    # Define model
    model = Sequential(
        [
            layers.Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_PADDING_LENGTH),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation=activations.relu),
            layers.Dense(6, activation=activations.softmax)
        ]
    )

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    # Train model
    history = model.fit(
        train_padded_sequences,
        train_labels_sequences,
        validation_data=(validation_padded_sequences, validation_labels_sequences),
        epochs=50,
        verbose=1
    )

    # Plot results
    plot_visualize_training(history)

    # Preparing files for projector.tensorflow.org
    save_vocabulary(filename=os.path.join(os.getcwd(), 'datasets', 'bbc_meta.tsv'),
                    vocabulary=sentence_tokenizer.word_index)

    save_embeddings(filename=os.path.join(os.getcwd(), 'datasets', 'bbc_vectors.tsv'),
                    embeddings=model.layers[0])


if __name__ == '__main__':
    main()
