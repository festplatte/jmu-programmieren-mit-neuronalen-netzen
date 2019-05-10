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

model = tf.keras.models.Sequential([
    # params: 6 * 5 * 5 + 6
    tf.keras.layers.Conv2D(6, (5, 5), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    # params: 16 * 3 * 3 * 6 + 16
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Flatten(),
    # 5 * 5 * 16 * 120 + 120
    tf.keras.layers.Dense(120, activation=tf.nn.relu),
    # 120 * 84 + 84
    tf.keras.layers.Dense(84, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    # 84 * 10 + 10
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
model.save('mnist-cnn.h5')
