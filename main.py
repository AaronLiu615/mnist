import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

#loading MNIST dataset from local file
with np.load('./mnist.npz') as data:
    x_train, y_train = data['x_train'], data['y_train']
    x_test, y_test = data['x_test'], data['y_test']

#normalize data to 0 to 1
x_train = x_train/255
x_test = x_test/255

model_lr = tf.keras.models.Sequential([
    tf.keras.layers.Input(x_train.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_lr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lr.summary()

#results from every case
model_lr.evaluate(x_test,y_test)

#results from set case with images
probs = model_lr.predict(x_test[:5])
pred = np.argmax(probs, axis=1)
for i in range(0):
    print(probs[i], " => ", pred[i])
    plt.imshow(x_test[i], cmap="Greys")
    plt.show()

#results from single case with images
probs = model_lr.predict(x_test[18:19])