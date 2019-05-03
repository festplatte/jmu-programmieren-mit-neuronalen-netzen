import numpy
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.load_model('mnist-tutorial.h5')

model.evaluate(x_test, y_test)

# see c)
predicted_classes = model.predict(x_test)
success = 0
for i in range(0,len(y_test)):
    if numpy.argmax(predicted_classes[i]) == y_test[i]:
        success = success + 1

accuracy = success / len(y_test)
print(accuracy)

# see d)
class_accuracy = [0,0,0,0,0,0,0,0,0,0]
class_count = [0,0,0,0,0,0,0,0,0,0]
for i in range(0,len(y_test)):
    class_count[y_test[i]] = class_count[y_test[i]] + 1
    if numpy.argmax(predicted_classes[i]) == y_test[i]:
        class_accuracy[y_test[i]] = class_accuracy[y_test[i]] + 1
    i = i + 1
for i in range(0,len(class_accuracy)):
    class_accuracy[i] = class_accuracy[i] / class_count[i]
print(class_accuracy)
