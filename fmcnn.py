import tensorflow as tf
import matplotlib.pyplot as plt
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            a=logs.get('accuracy')
            if(a>0.70):
                print("\nReached 99.8% accuracy so cancelling training!")
                self.model.stop_training=True
    # YOUR CODE ENDS HERE
    callbacks=myCallback()
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # YOUR CODE STARTS HERE
    training_images=training_images/255.0
    training_images=training_images.reshape(60000, 28, 28, 1)
    test_images=test_images/255.0
    test_images=test_images.reshape(10000, 28, 28, 1)
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(64,(3,3), activation='relu',input_shape=(28,28,1)),
                                        tf.keras.layers.MaxPooling2D(2,2),
                                        tf.keras.layers.Flatten(),
                                        tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
          

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # model fitting
    history = model.fit(training_images,training_labels, epochs=20, callbacks=[callbacks])
    # model fitting
    loss = history.history['loss']
    epochs = range(1,2)
    plt.plot(epochs, loss, 'g', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    model.summary();
    
    
    return history.epoch, history.history['accuracy'][-1]

_, _ = train_mnist_conv()

