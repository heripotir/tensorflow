#python mnist_tensorflow_trial_pne.py

import tensorflow as tf
import numpy
mnist= tf.keras.datasets.mnist

#loading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocessing
x_train=x_train.reshape(60000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)


