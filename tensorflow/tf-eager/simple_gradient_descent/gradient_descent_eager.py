#!/usr/bin/env python
import os
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()

NUM_EXAMPLES = 2000

training_inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
outputs = training_inputs * 3 + 2 + noise


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.W = tfe.Variable(5., name="weight")
        self.b = tfe.Variable(0., name="bias")

    def predict(self, input):
        return self.W * input + self.b


def loss(model, inputs, outputs):
    error = model.predict(inputs) - outputs
    return tf.reduce_mean(tf.square(error))


def grad(model, inputs, outputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, outputs)
    return tape.gradient(loss_value, [model.W, model.b])


if __name__ == "__main__":
    model = Model()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    global_step = tf.train.get_or_create_global_step()

    writer = tf.contrib.summary.create_file_writer("logs")
    writer.set_as_default()

    print("Initial loss: {:.3f}".format(loss(model, training_inputs, outputs)))

    for i in range(300):
        gradients = grad(model, training_inputs, outputs)
        optimizer.apply_gradients(zip(gradients, [model.W, model.b]),
                                  global_step=global_step)

        with tf.contrib.summary.record_summaries_every_n_global_steps(5):
            tf.contrib.summary.scalar('loss', loss(model,
                                                   training_inputs,
                                                   outputs))

        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(model,
                                                               training_inputs,
                                                               outputs)))

    print("Final loss: {:.3f}".format(loss(model,
                                           training_inputs,
                                           outputs)))

    print("W = {}, B = {}".format(model.W.numpy(), model.b.numpy()))

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    root = tfe.Checkpoint(optimizer=optimizer,
                          model=model,
                          optimizer_step=tf.train.get_or_create_global_step())
    root.save(file_prefix=checkpoint_prefix)
