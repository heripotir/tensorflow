import tensorflow as tf
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            a=logs.get('accuracy')
            if(a>0.998):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training=True

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# YOUR CODE STARTS HERE
callbacks=myCallback()
training_images=training_images/255.0
training_images=training_images.reshape(60000, 28, 28, 1)
test_images=test_images/255.0
test_images=test_images.reshape(10000, 28, 28, 1)
# YOUR CODE ENDS HERE

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3), activation='relu',input_shape=(28,28,1)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(training_images,training_labels, epochs=10, callbacks=[callbacks])

ac = history.history['accuracy']
epochs = range(1,11)
plt.plot(epochs, ac, 'g', label='Training Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
model.summary();



model.evaluate(test_images, test_labels)
