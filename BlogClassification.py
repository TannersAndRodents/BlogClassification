#!/usr/bin/env python
"""Does a text classification of a dataset of blog posts by authors in different age groups.
Dataset: https://u.cs.biu.ac.il/~koppel/BlogCorpus.htm
J. Schler, M. Koppel, S. Argamon and J. Pennebaker (2006). Effects of Age and Gender on Blogging in Proceedings of 2006 AAAI Spring Symposium on Computational Approaches for Analyzing Weblogs. 
"""
from tensorflow import keras
import tensorflow as tf
import os

def main():

    # Load data

    # VS Code runs from project root. If run from bash, BlogClassification is root folder.
    # Check which option aplies and adapt relative paths
    if os.path.exists("preparedDataset/"):
        path_to_ds = "preparedDataset/"
    elif os.path.exists("BlogClassification/preparedDataset/"):
        path_to_ds = "BlogClassification/preparedDataset/"
    else:
        raise FileNotFoundError

    training_data_path = path_to_ds + "training_ds"
    test_data_path =  path_to_ds + "test_ds"
    val_data_path = path_to_ds + "validation_ds"

    print("path: ", training_data_path)

    batch_size = 32
    raw_train_ds = keras.utils.text_dataset_from_directory(
        training_data_path,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=1337
    )

    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        val_data_path,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=1337
    )

    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        test_data_path,
        batch_size=batch_size,
    )

    # Prepare and vectorize text

    # Model constants
    max_features = 20000
    embedding_dim = 128
    sequence_length = 500

    vectorize_layer = keras.layers.TextVectorization(
        max_tokens=max_features,
        standardize="lower",
        output_mode="int",
        output_sequence_length=sequence_length
    )

    # Using the full training dataset to adapt TextVectorization
    # Ineffizcient I guess (Only a portion should sufficide) but why not
    adapt_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(adapt_ds)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # caching for better performance
    train_ds = train_ds.cache().prefetch(buffer_size=20)
    val_ds = val_ds.cache().prefetch(buffer_size=20)
    test_ds = test_ds.cache().prefetch(buffer_size=20)

    # print some elements from dataset to examine
    for text_batch, label_batch in raw_train_ds.take(3):
        for i in range(5):
            print(text_batch.numpy()[i])
            print(label_batch.numpy()[i])


     # Model
    inputs = tf.keras.Input(shape=(None,), dtype="int64")

    x = keras.layers.Embedding(max_features, embedding_dim)(inputs)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = keras.layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)

    x = keras.layers.GlobalMaxPool1D()(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    predictions = keras.layers.Dense(1, activation="sigmoid", name="predictions")(x)

    model = tf.keras.Model(inputs, predictions)

    # compile model
    # loss and optimizer might not be optimal
    model.compile(loss="binary_crossentropy", optimizer="adagrad", metrics=["accuracy"])

    # Train model
    epochs = 1
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)  

    # Test model on test dataset
    model.evaluate(test_ds)

    return

if __name__ ==  "__main__":
    main()
