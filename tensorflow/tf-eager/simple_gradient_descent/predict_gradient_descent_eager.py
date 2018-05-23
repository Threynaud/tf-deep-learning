#!/usr/bin/env python
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from gradient_descent_eager import Model

if __name__ == "__main__":
    model = Model()
    checkpointer = tfe.Checkpoint(abc=model)
    checkpointer.restore(tf.train.latest_checkpoint('checkpoints/'))
    print(model.predict(7))
