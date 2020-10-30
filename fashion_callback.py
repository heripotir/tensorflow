import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        #if (logs.get('loss')<0.3):
        if(logs.get('accuracy')>0.9):
            #or instead if we wanted to work with accuracy, we could write if(logs.get('accuracy')>0.6):
            print("\nLoss is low so cancelling training!!")
            self.model.stop_training=True


callbacks=myCallback()
mnist=tf.keras.datasets.fashion_mnist
print('data loading..')
(training_images, training_labels),(test_images,test_labels)=mnist.load_data()
print('data loaded..')
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])
training_images = training_images/255.0 
test_images = test_images/255.0

model=tf.keras.models.Sequential([tf.keras.layers.Flatten(),tf.keras.layers.Dense(128, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer=tf.optimizers.Adam(), loss= 'sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels, epochs=10, callbacks=[callbacks])
model.evaluate(test_images, test_labels)

model.summary()


