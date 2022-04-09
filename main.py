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
from sklearn.metrics import precision_recall_fscore_support as score
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import tensorflow_hub as hub
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sn
import os


# Global vars:
EPOCHS = 100
PATIENCE = 6
RANDOM_SEED = 420
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
    samples = 2000
    test_val_samples = 250

    # Let's get rid of categories with too few records or ambiguous labels:
    invalid_cols = [
        "EDUCATION",
        "CULTURE & ARTS",
        "WEIRD NEWS",
        "FIFTY",
        "IMPACT",
        "LATINO VOICES",
        "COLLEGE",
        "GOOD NEWS",
        "TECH",
        "SCIENCE"
    ]
    df = df.loc[~df.category.isin(invalid_cols)]

    # Let's map similar categories together:
    df.loc[df.category == 'ARTS', 'category'] = 'ARTS & CULTURE'
    df.loc[df.category == 'MONEY', 'category'] = 'BUSINESS'
    df.loc[df.category == 'TASTE', 'category'] = 'FOOD & DRINK'
    df.loc[df.category == 'GREEN', 'category'] = 'ENVIRONMENT'
    df.loc[df.category == 'HEALTHY LIVING', 'category'] = 'WELLNESS'
    df.loc[df.category == 'PARENTS', 'category'] = 'PARENTING'
    df.loc[df.category == 'STYLE', 'category'] = 'STYLE & BEAUTY'
    df.loc[df.category == 'THE WORLDPOST', 'category'] = 'WORLD NEWS'
    df.loc[df.category == 'WORLDPOST', 'category'] = 'WORLD NEWS'

    # Let's encode the categories into integers:
    cats = sorted(df.category.unique())
    df.category = df.category.apply(cats.index)

    # Let's assemble the datasets:
    X_train_index = get_train_samples(df, samples)
    X_train = df.loc[X_train_index]
    X_test = df.loc[~df.index.isin(X_train_index)]

    # From the Test dataset, let's break it down using a 50/50 split:
    X_test_index = get_train_samples(X_test, test_val_samples)
    X_val = X_test.loc[X_test_index]
    X_test = X_test.loc[~X_test.index.isin(X_test_index)]
    X_test = X_test.loc[get_train_samples(X_test, test_val_samples)]

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.headline.values, X_train.category.values))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.headline.values, X_val.category.values))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test.headline.values, X_test.category.values))
    return train_dataset, val_dataset, test_dataset, cats


def build_custom_network(outputs: int) -> models.Sequential:
    """ Build a sequential model using a pretrained embedding layer from TFHub. """
    # TFHub Sources:
    emb_layer = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128-with-normalization/1"

    # Assemble the model:
    lyrs = [
        hub.KerasLayer(emb_layer, input_shape=[], dtype=tf.string),
        # layers.Dense(50, activation='relu'),
        # layers.Dropout(0.25),
        # layers.Dense(50, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(outputs, activation='softmax')
    ]
    model = models.Sequential(name='PreTrainedEmbed-SNN', layers=lyrs)

    # Compile it and return it:
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics='accuracy'
    )
    return model


def train_pretrained_network(training_ds, validation_ds, labels, pretrain_rounds=10) -> models.Sequential:
    """ Build a model from a pretrained one. Freeze the bottom layers and train the top layers
    for n epochs. Then, unfreeze the bottom layers, adjust the learning rate down and train the
    model one more time. """
    # Start pretraining:
    model = build_custom_network(len(labels))
    version_name = get_model_version_name(model.name)
    tb_logs = callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    model.fit(training_ds, validation_data=validation_ds, epochs=pretrain_rounds, callbacks=[tb_logs])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))

    # Unfreeze layers and train the entire network:
    for layer in model.layers:
        layer.trainable = True
    model.compile(
        loss=losses.SparseCategoricalCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics='accuracy'
    )
    early_stop = callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(training_ds, validation_data=validation_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return model


def plot_confision_matrix(model, test_dataset, version_name, label_names):
    # Fetch predictions and true labels:
    predictions = []
    labels = []
    for x, y in tqdm(test_dataset, desc='Predicting labels on test dataset'):
        predictions += list(tf.argmax(model.predict(x), axis=1).numpy().astype(float))
        labels += list(y.numpy().astype(float))

    # Build a confusion matrix and save the plot in a PNG file:
    matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions).numpy()
    df = pd.DataFrame(matrix)
    df.columns = label_names
    df.index = label_names
    sn.set(font_scale=0.5)
    cf = sn.heatmap(df, annot=True, fmt="d")
    cf.set(xlabel='Actuals', ylabel='Predicted')
    cf.get_figure().savefig(os.path.join(CM_DIR, version_name.replace('.h5', '.png')))

    # Compute precision and recall:
    precision, recall, f1, _ = score(labels, predictions)
    print(f"Model Test Accuracy: {model.evaluate(test_dataset)}")
    print(f"Precision: {np.mean(precision)}")
    print(f"Recall: {np.mean(recall)}")
    print(f"F1 Score: {np.mean(f1)}")


def testing():
    # Get datasets and perform preprocessing:
    X_train, X_val, X_test, labels = get_datasets()
    X_test = X_test.cache().shuffle(10_000).batch(32).prefetch(buffer_size=AUTOTUNE)

    # Load a trained model for evaluation:
    trained_models = os.listdir(r'models')
    selected_model = trained_models[-1]
    model = models.load_model(f"models/{selected_model}", custom_objects={'KerasLayer': hub.KerasLayer})
    return plot_confision_matrix(model, X_test, selected_model, labels)


def main():
    """ Run script. """
    # Get datasets and perform preprocessing:
    X_train, X_val, X_test, labels = get_datasets()
    X_train = X_train.cache().shuffle(10_000).batch(32).prefetch(buffer_size=AUTOTUNE)
    X_val = X_val.cache().shuffle(10_000).batch(32).prefetch(buffer_size=AUTOTUNE)
    X_test = X_test.cache().shuffle(10_000).batch(32).prefetch(buffer_size=AUTOTUNE)

    #  Start training and evaluation afterwards:
    model = train_pretrained_network(X_train, X_val, labels)
    trained_models = os.listdir(r'models')
    selected_model = trained_models[-1]
    return plot_confision_matrix(model, X_test, selected_model, labels)


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    pd.set_option('expand_frame_repr', False)
    main()
