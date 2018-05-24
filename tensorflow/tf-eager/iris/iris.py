#!/usr/bin/env python
import os

import click
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from sklearn import datasets, model_selection, preprocessing

# TODO: Refactor summaries writing, it's a mess

tf.enable_eager_execution()


def import_data():
    data = datasets.load_iris()

    X = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit_transform(
        data['data'])

    y = preprocessing.OneHotEncoder(sparse=False).fit_transform(
        data['target'].reshape(-1, 1))

    X_train, X_dev, y_train, y_dev = model_selection.train_test_split(
        X, y, test_size=0.25, stratify=y)
    return X_train, X_dev, y_train, y_dev


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self._hlayer = tf.layers.Dense(10, activation=tf.nn.sigmoid)
        self._olayer = tf.layers.Dense(3, activation=None)

    def call(self, inputs):
        outputs = self._olayer(self._hlayer(inputs))
        return outputs


def create_dataset(X, y, num_epochs=1, batch_size=10):
    dataset = (tf.data.Dataset().from_tensor_slices((X, y))
                                .shuffle(1000)
                                .repeat(num_epochs)
                                .batch(batch_size))
    # iterator = tfe.Iterator(dataset)
    return dataset


def sce(model, inputs, labels):
    logits = model(inputs)
    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=labels))
    return(loss)


def evaluate(model, dataset):
    avg_loss = tfe.metrics.Mean('loss')
    accuracy = tfe.metrics.Accuracy('accuracy')
    iterator = tfe.Iterator(dataset)

    for inputs, labels in iterator:
        avg_loss(sce(model, inputs, labels))
        accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int64),
                 tf.argmax(labels, axis=1))
    print(f"Dev set: Average loss: {avg_loss.result()},\
    Accuracy: {100 * accuracy.result()}")

    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', avg_loss.result())
        tf.contrib.summary.scalar('accuracy', accuracy.result())


# TODO: Make histograms great again
def save_variables_histogram(model):
    tf.contrib.summary.histogram(
        "hidden/weight", tf.cast(model.variables[0], tf.float32))
    tf.contrib.summary.histogram(
        "hidden/bias", tf.cast(model.variables[1], tf.float32))
    tf.contrib.summary.histogram(
        "hidden/weight", tf.cast(model.variables[2], tf.float32))
    tf.contrib.summary.histogram(
        "hidden/bias", tf.cast(model.variables[3], tf.float32))


def fit(model,
        training_dataset,
        dev_dataset,
        optimizer,
        verbose=False,
        logdir=None,
        summary_freq=3,
        eval_freq=5):

    loss_and_grads = tfe.implicit_value_and_gradients(sce)
    tf.train.get_or_create_global_step()

    if logdir:
        if tf.gfile.Exists(logdir):
            tf.gfile.DeleteRecursively(logdir)
        tf.gfile.MakeDirs(logdir)
        train_dir = os.path.join(logdir, 'train')
        dev_dir = os.path.join(logdir, 'dev')
    else:
        train_dir = None
        dev_dir = None
    summary_writer = tf.contrib.summary.create_file_writer(
        train_dir, flush_millis=10000)
    dev_summary_writer = tf.contrib.summary.create_file_writer(
        dev_dir, flush_millis=10000, name='dev')
    # Note: Important to give a name to at least one of them or tensorfboard
    # mix them up for no reason

    for batched_inputs, batched_labels in tfe.Iterator(training_dataset):
        with summary_writer.as_default():
            loss, grads = loss_and_grads(
                model, batched_inputs, batched_labels)
            optimizer.apply_gradients(
                grads, global_step=tf.train.get_global_step())

            global_step = tf.train.get_global_step().numpy()
            if verbose and global_step % 10 == 0:
                print(f"Loss at step {global_step}: {loss}")

            if logdir:
                with tf.contrib.summary.record_summaries_every_n_global_steps(summary_freq):  # NOQA
                    tf.contrib.summary.scalar("loss", loss)
                    save_variables_histogram(model)

        if logdir and global_step % eval_freq == 0:
            with dev_summary_writer.as_default():
                evaluate(model, dev_dataset)


@click.command()
@click.option("--num-epochs", "-n", default=1, type=int,
              help="Number of epochs")
@click.option("--batch_size", "-b", default=1, type=int,
              help="Batch size")
@click.option("--verbose", "-v", is_flag=True, help="Activate verbose mode")
@click.option("--logdir", "-l", type=str, help="Logdir")
@click.option("--summary-freq", "-f", default=1, type=int,
              help="Frequency at which summary are recorded (number of steps)")
@click.option("--eval-freq", "-e", default=3, type=int,
              help="Frequency at which model is evaluated (number of steps)")
def train_iris_model(num_epochs, batch_size, verbose,
                     logdir, summary_freq, eval_freq):
    X_train, X_dev, y_train, y_dev = import_data()
    training_dataset = create_dataset(X_train,
                                      y_train,
                                      num_epochs=num_epochs,
                                      batch_size=batch_size)
    dev_dataset = create_dataset(X_dev,
                                 y_dev,
                                 num_epochs=1,
                                 batch_size=1)

    model = Model()

    device = "gpu:0" if tfe.num_gpus() else "cpu:0"
    print(f"Using device: {device}")
    with tf.device(device):
        optimizer = tf.train.AdamOptimizer()
        fit(model, training_dataset, dev_dataset, optimizer, verbose=verbose,
            logdir=logdir, summary_freq=summary_freq)


if __name__ == "__main__":
    train_iris_model()
