import ray
from ray import tune
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from ray.tune import Trainable
from ray.tune.integration.keras import TuneReportCallback
import os
import random
import numpy as np
from filelock import FileLock

def objective(config):
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    with FileLock(os.path.expanduser("~/.data.lock")):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=777)
    seed =777
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    inputs = Input(shape=(28,28))
    flattens = Flatten()(inputs)
    hidden1 = Dense(config["neuron1"], activation=config["activation"])(flattens)
    hidden2 = Dense(config["neuron2"], activation=config["activation"])(hidden1)
    output = Dense(10, activation='softmax')(hidden2)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=config["optimizers"], loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, verbose=0, epochs=10, shuffle=False, callbacks=[TuneReportCallback({
            "mean_accuracy": "accuracy"
        })])

