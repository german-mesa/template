import os

import tensorflow as tf

from tensorflow.keras import utils


def get_tweet_dataset(url, batch_size=10):
    filename = utils.get_file(fname=os.path.basename(url),
                              origin=url,
                              cache_dir='./core')

    dataset = tf.data.experimental.CsvDataset(
        filenames=filename,
        record_defaults=[int(), str()],
        field_delim=',',
        header=True,
        select_cols=[0, 5]
    )

    dataset = dataset.map(lambda *x: (x[0], x[1]))
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


if __name__ == '__main__':
    # Get data from CSV file
    corpus = get_tweet_dataset("https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv")

    for item in corpus:
        print(item[0].numpy())
        print(item[1].numpy())
