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

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

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

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_function(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss(loss)
        train_acc(y, predictions)
    
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        loss = loss_function(labels, predictions)
        
        test_loss(loss)
        test_acc(labels, predictions)

    for epoch in range(config["training_iteration"]):
        for images, labels in train_ds:
            train_step(images, labels)
            
        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)
            
        tune.report(mean_accuracy=test_acc.result().numpy().item())