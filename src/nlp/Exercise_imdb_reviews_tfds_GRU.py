import tensorflow_datasets as tfds

import tensorflow as tf
import matplotlib.pyplot as plt

NUM_EPOCHS = 10
BUFFER_SIZE = 10000
BATCH_SIZE = 64
MODEL_SELECTED = 'GRU'

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


def main():
    # Get the data - PrefetchDataset and tfds.core.DatasetInfo
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    # Prepare dataset
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
    test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))

    # Tokenizer from dataset
    tokenizer = info.features['text'].encoder

    # Preparing our new model
    if MODEL_SELECTED == 'GRU':
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
                tf.keras.layers.Dense(6, activation=tf.keras.activations.relu),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
            ]
        )
    else:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
                tf.keras.layers.Conv1D(128, 5, activation=tf.keras.activations.relu),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
                tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
            ]
        )

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=test_dataset
    )

    # Plot results
    plot_graphs(history, 'accuracy')
    plot_graphs(history, 'loss')


if __name__ == '__main__':
    main()
