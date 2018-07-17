#!/usr/bin/env python
import os

import click
import tensorflow as tf
from sklearn import datasets, model_selection, preprocessing


def import_data():
    data = datasets.load_iris()

    X = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit_transform(
        data['data'])

    y = preprocessing.OneHotEncoder(sparse=False).fit_transform(
        data['target'].reshape(-1, 1))

    X_train, X_dev, y_train, y_dev = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    feature_names = list(map(
        lambda x: '_'.join(x.split()[:2]), data['feature_names']))

    return {"data": (X_train, X_dev, y_train, y_dev),
            "feature_names": feature_names}


def train_input_fn(X, y, num_epochs=1, batch_size=1, shuffle=False):
    dataset = (tf.data.Dataset().from_tensor_slices((X, y))
                                .shuffle(1000)
                                .repeat(num_epochs)
                                .batch(batch_size))
    iterator = dataset.make_one_shot_iterator()
    X, y = iterator.get_next()
    output = ({'iris_spec': X}, y)
    return output


data = import_data()['data']
with tf.Session() as sess:
    for _ in range(10):
        print(train_input_fn(data[0], data[2])[0]['iris_spec'].eval())
