"""
Title: main.py

Created on: 3/29/2022

Author: FriscianViales

Encoding: UTF-8

Description: Train a neural network to classify news types based on headlines.
"""


from tqdm import tqdm
from datetime import datetime
from pytz import timezone
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# Global vars:
EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 69
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
CM_DIR = os.path.join(os.getcwd(), 'confusion-matrixes')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
AUTOTUNE = tf.data.AUTOTUNE


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def read_data() -> pd.DataFrame:
    """ Read raw JSON file into a pandas df. """
    file_path = 'data/News_Category_Dataset_v2.json'
    return pd.read_json(file_path, lines=True)


def get_train_samples(df: pd.DataFrame, total_samples: int) -> list:
    """ Return an index based on random samples for each category in the dataset. """
    # Let's build a dict with an index per category to filter over records:
    indexes = dict()
    for x in df.category.unique():
        indexes[x] = set(df.loc[df.category == x].index)

    # Let's sample some rows for each category:
    X_train_index = []
    for category in tqdm(indexes.keys(), desc="Sampling categories"):
        for row in range(0, total_samples):
            samples = indexes.get(category)
            random_sample = np.random.choice(list(samples))
            samples.discard(random_sample)
            indexes[category] = samples
            X_train_index.append(random_sample)

    return X_train_index


def get_datasets() -> tuple:
    """ Read and transform raw data. After that, it returns a tuple with the training and testing dataset. """
    # Read data and filter it down:
    df = read_data()
    df = df[["headline", "category"]]

    # Let's get rid of categories with too few records:
    invalid_cols = ["EDUCATION", "CULTURE & ARTS"]
    df = df.loc[~df.category.isin(invalid_cols)]

    # Let's encode the categories into integers:
    cats = sorted(df.category.unique())
    df.category = df.category.apply(cats.index)

    # Let's assemble the datasets:
    X_train_index = get_train_samples(df, 1000)
    X_train = df.loc[X_train_index]
    X_test = df.loc[~df.index.isin(X_train_index)]

    # From the Test dataset, let's break it down using a 50/50 split:
    X_test_index = get_train_samples(X_test, 50)
    X_val = X_test.loc[X_test_index]
    X_test = X_test.loc[~X_test.index.isin(X_test_index)]
    X_test = X_test.loc[get_train_samples(X_test, 50)]

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.headline.values, X_train.category.values))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.headline.values, X_val.category.values))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.headline.values, X_test.category.values))
    return train_dataset, val_dataset, test_dataset, cats


def build_custom_network() -> models.Sequential:
    """ Build a sequential model using a pretrained embedding layer from TFHub. """
    # TFHub Sources:
    emb_layer = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"

    # Assemble the model:
    lyrs = [
        hub.KerasLayer(emb_layer, input_shape=[], dtype=tf.string),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1)
    ]
    model = models.Sequential(name='PreTrainedEmbed-SNN', layers=lyrs)

    # Compile it and return it:
    model.compile(
        loss=losses.BinaryCrossentropy(from_logits=True),
        optimizer=optimizers.Adam(),
        metrics=metrics.BinaryAccuracy()
    )
    return model


def train_pretrained_network(training_ds, validation_ds, pretrain_rounds=10) -> models.Sequential:
    """ Build a model from a pretrained one. Freeze the bottom layers and train the top layers
    for n epochs. Then, unfreeze the bottom layers, adjust the learning rate down and train the
    model one more time. """
    # Start pretraining:
    model = build_custom_network()
    version_name = get_model_version_name(model.name)
    tb_logs = callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    model.fit(training_ds, validation_data=validation_ds, epochs=pretrain_rounds, callbacks=[tb_logs])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Unfreeze layers and train the entire network:
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        loss=losses.BinaryCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=metrics.BinaryAccuracy()
    )
    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(training_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return model


def main():
    """ Run script. """
    # Get datasets and perform preprocessing:
    X_train, X_val, X_test, labels = get_datasets()
    X_train = X_train.cache().shuffle(10_000).batch(32).prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().shuffle(10_000).batch(32).prefetch(buffer_size=AUTOTUNE)
    model = train_pretrained_network(X_train, X_val)
    print(model)
    return {}


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    pd.set_option('expand_frame_repr', False)
    main()
