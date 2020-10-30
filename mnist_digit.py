
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            a=logs.get('accuracy')
            if(a>0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training=True
callbacks=myCallback()
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                   tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                   tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

x_train=np.array(x_train)
y_train=np.array(y_train)
history = model.fit(x_train,y_train, epochs=10, callbacks=[callbacks])
model.evaluate(x_test, y_test)
