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

print(x_train.shape)
print(y_train.shape)

# plt.imshow(x_train[0], cmap='Greys')
# plt.show()

#training using onehot from pytorch
model_lr = tf.keras.models.Sequential([
    tf.keras.layers.Input(x_train.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_lr.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lr.summary()

y_onehot_train = tf.one_hot(y_train, 10)
model_lr.fit(x_train, y_onehot_train)


#training using sparse loss
model_lr = tf.keras.models.Sequential([
    tf.keras.layers.Input(x_train.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model_lr.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_lr.summary()

#random sample
model_lr.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

#given sample
model_lr.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))

#visual graphs
history_lr = model_lr.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test), verbose=False)

plt.plot(history_lr.history['loss'], label='train') 
plt.plot(history_lr.history['val_loss'], label='val')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history_lr.history['accuracy'], label='train')
plt.plot(history_lr.history['val_accuracy'], label='val')
plt.ylabel('accuracy')
plt.legend()
plt.show()