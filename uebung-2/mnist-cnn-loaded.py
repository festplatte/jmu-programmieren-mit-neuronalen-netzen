import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

plt.imshow(x_train[2], cmap='gray')
plt.show()

x_train = np.expand_dims(x_train, 3)
x_test = np.expand_dims(x_test, 3)

model = tf.keras.models.load_model('mnist-cnn.h5')

print(model.summary())
